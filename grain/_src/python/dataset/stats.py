# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tools for recording statistics about dataset transformations."""

from __future__ import annotations

import abc
import contextlib
import pprint
import sys
import threading
import time
import types
from typing import Sequence, TypeVar

from absl import logging
from grain._src.core import config
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import tree

from grain._src.core import monitoring


_self_time_ms_histogram = monitoring.EventMetric(
    "/grain/python/dataset/self_time_ms",
    metadata=monitoring.Metadata(
        description=(
            "Histogram of transformation self time. Each data point is the "
            "average value of self times/element produced during a monitoring "
            "interval."
        ),
        units=monitoring.Units.MILLISECONDS,
    ),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
    bucketer=monitoring.Bucketer.PowersOf(2.0),
)

T = TypeVar("T")
# Time between two consecutive monitoring reports.
_REPORTING_PERIOD_SEC = 10
_LOG_EXECTION_SUMMARY_PERIOD_SEC = 60

_EDGE_TEMPLATE = r"""{input_spec}
  ││
  ││  {transform}
  ││
  ╲╱
{output_spec}
"""

_AVG_PROCESSING_TIME_COLUMN_NAME = "avg processing time"

_COLUMN_NAME_OVERRIDES = types.MappingProxyType({
    "min_processing_time_ns": "min processing time",
    "max_processing_time_ns": "max processing time",
    "total_processing_time_ns": "total processing time",
    "num_produced_elements": "num produced elements",
    "output_spec": "output spec",
})


def _pretty_format_ns(value: int) -> str:
  """Pretty formats a time value in nanoseconds to human readable value."""
  if value < 1000:
    return f"{value}ns"
  elif value < 1000_000:
    return f"{value/1000:.2f}us"
  elif value < 1_000_000_000:
    return f"{value/1000_000:.2f}ms"
  else:
    return f"{value/1000_000_000:.2f}s"


def _get_avg_processing_time_ns(
    summary: execution_summary_pb2.ExecutionSummary,
    node_id: int,
) -> int:
  """Returns the average processing time in nanoseconds for the given node."""
  if summary.nodes[node_id].num_produced_elements == 0:
    return 0
  return int(
      summary.nodes[node_id].total_processing_time_ns
      / summary.nodes[node_id].num_produced_elements
  )


def _pretty_format_summary(
    summary: execution_summary_pb2.ExecutionSummary,
) -> str:
  """Returns Execution Stats Summary for the dataset pipeline in tabular format."""
  tabular_summary = []
  col_names = [key for key in summary.nodes[0].DESCRIPTOR.fields_by_name.keys()]
  # Insert the average processing time column after the max processing time
  # column.
  index = col_names.index("max_processing_time_ns")
  col_names.insert(index + 1, _AVG_PROCESSING_TIME_COLUMN_NAME)

  # Compute the maximum width of each column.
  col_widths = []
  for name in col_names:
    max_width = len(_COLUMN_NAME_OVERRIDES.get(name, name))
    for node_id in summary.nodes:
      if name == _AVG_PROCESSING_TIME_COLUMN_NAME:
        value = _get_avg_processing_time_ns(summary, node_id)
      else:
        value = getattr(summary.nodes[node_id], name)
      max_width = max(len(str(value)), max_width)
    col_widths.append(max_width)

  col_headers = [
      "| {:<{}} |".format(_COLUMN_NAME_OVERRIDES.get(name, name), width)
      for name, width in zip(col_names, col_widths)
  ]
  col_seperators = ["-" * len(header) for header in col_headers]

  tabular_summary.extend(col_seperators)
  tabular_summary.append("\n")
  tabular_summary.extend(col_headers)
  tabular_summary.append("\n")
  tabular_summary.extend(col_seperators)
  tabular_summary.append("\n")

  for node_id in sorted(summary.nodes, reverse=True):
    is_total_processing_time_zero = (
        summary.nodes[node_id].total_processing_time_ns == 0
    )
    for name, width in zip(col_names, col_widths):
      if name == _AVG_PROCESSING_TIME_COLUMN_NAME:
        value = _get_avg_processing_time_ns(summary, node_id)
      else:
        value = getattr(summary.nodes[node_id], name)

      if name in (
          "min_processing_time_ns",
          "max_processing_time_ns",
          "total_processing_time_ns",
          _AVG_PROCESSING_TIME_COLUMN_NAME,
          "num_produced_elements",
      ):
        # If the total processing time is zero, the pipeline has not yet
        # produced an element and processing times & num_produced_elements are
        # not yet meaningful.
        if is_total_processing_time_zero:
          col_value = f"{f'| N/A':<{width+2}} |"
        elif name != "num_produced_elements":
          col_value = f"{f'| {_pretty_format_ns(value)}':<{width+2}} |"
        else:
          col_value = f"{f'| {value}':<{width+2}} |"
      else:
        col_value = "| {:<{}} |".format(str(value), width)
      tabular_summary.append(col_value)
    tabular_summary.append("\n")

  for seperator in col_seperators:
    tabular_summary.append(seperator)
  return "".join(tabular_summary)


class Timer:
  """Context manager to time blocks of code.

  The value is accumulated across multiple usages as a context manager. Expected
  to be used as show below. Note that `Timer` is not thread-safe and is intended
  to be used as a local variable.
  ```
    timer = Timer()
    with timer:
      <code block 1>
    with timer:
      <code block 2>
    self_time = timer.value()
  ```
  """

  def __init__(self):
    self._accumulator = 0
    self._last = 0

  def __enter__(self):
    self._last = time.perf_counter_ns()

  def __exit__(self, *args):
    self._accumulator += time.perf_counter_ns() - self._last

  def value(self):
    """Returns the accumulated timer value across multiple usages."""
    return self._accumulator

  def reset(self):
    """Resets the accumulated timer value to 0."""
    self._accumulator = 0
    self._last = 0


class Stats(abc.ABC):
  """Base abstract class for statistics recording.

  This class replicates the transformation tree structure and provides
  interfaces for recording statistics in the given transformation node.
  """

  def __init__(self, name: str, parents: Sequence[Stats]):
    self.name = name
    self._output_spec = None
    self._parents = parents
    # Mark parent nodes as non-outputs. Nodes that are not updated are the
    # output nodes.
    self._is_output = True
    for p in parents:
      p._is_output = False

  @contextlib.contextmanager
  @abc.abstractmethod
  def record_self_time(self, offset_ns: float = 0, num_produced_elements=1):
    """Records time spent in this node's transfromation.

    Thread-safe.

    Implemented as context manager for convenience. Expected to be used as
    follows:
    ```
    class MyMapDataset(MapDataset):
      ...
      def __getitem__(self, index):
        input_element = self._parent[index]
        with self._stats.record_self_time():
          return self._map_fn(input_element)
    ```
    and
    ```
    class MyMapDatasetIterator(DatasetIterator):
      ...
      def __next__(self):
        input_element = next(self._parent)
        with self._stats.record_self_time():
          return self._map_fn(input_element)
    ```

    Args:
      offset_ns: (Optional.) A offset to add to the self time measured by this
        function. Default to 0.
      num_produced_elements: (Optional) The number of elements produced during
        the measured self time. Default to 1.
    """
    ...

  @abc.abstractmethod
  def record_output_spec(self, element: T) -> T:
    """Records output spec of the elements produced by this node.

    Thread-safe.

    Args:
      element: structure to record the spec of.

    Returns: the `element` unchanged (for convenience).

    Expected to be used as follows:
    ```
    class MyMapDataset(MapDataset):
      ...
      def __getitem__(self, index):
        input_element = self._parent[index]
        return self._stats.record_output_spec(self._map_fn(input_element))
    ```
    and
    ```
    class MyMapDatasetIterator(DatasetIterator):
      ...
      def __next__(self):
        input_element = next(self._parent)
        return self._stats.record_output_spec(self._map_fn(input_element))
    ```
    """
    ...

  @abc.abstractmethod
  def report(self):
    """Reports the collected statistics.

    This should be expected to be called once the last element is processed as
    well as in the middle of execution.

    Not thread-safe, expected to be called from a single thread.
    """
    ...

  def _visualize_dataset_graph(self):
    """Generates Dataset visualization graph."""
    # TODO:Save the graph to a dot file for advanced visualization.
    has_multiple_parents, dataset_graph = self._build_visualization_str()
    if has_multiple_parents:
      dataset_graph = (
          "WARNING: Detected multi-parent datasets. Only displaying the first"
          " parent.\n\n"
          + dataset_graph
      )
    return dataset_graph

  def _build_visualization_str(self, has_multiple_parents: bool = False):
    """Builds Dataset visualization graph."""
    if self._parents:
      if len(self._parents) > 1:
        has_multiple_parents = True
      # pylint: disable=protected-access
      has_multiple_parents, parent_vis = self._parents[
          0
      ]._build_visualization_str(has_multiple_parents)
      # pylint: enable=protected-access
      transform_repr = self.name
    else:
      parent_vis = self.name
      transform_repr = ""
    return (
        has_multiple_parents,
        _EDGE_TEMPLATE.format(
            input_spec=parent_vis,
            transform=transform_repr,
            output_spec=pprint.pformat(self._output_spec),
        ),
    )


class _NoopStats(Stats):
  """Default implementation for statistics collection that does nothing."""

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0, num_produced_elements=1):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass


class _VisualizationStats(Stats):
  """Produces Dataset Visualization Graph."""

  def __init__(self, name: str, parents: Sequence[Stats]):
    super().__init__(name, parents)
    self._reported = False
    self._reported_lock = threading.Lock()

  def __reduce__(self):
    return _VisualizationStats, (self.name, self._parents)

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0, num_produced_elements=1):
    yield

  def record_output_spec(self, element: T) -> T:
    # Visualize the dataset graph once last node had seen a non-None element.
    if self._output_spec is None:
      self._output_spec = tree.spec_like(element)
      if self._is_output and not self._reported:
        # The check above with update without a lock is not atomic, need to
        # check again under a lock.
        with self._reported_lock:
          if not self._reported:
            self.report()
            self._reported = True
    return element

  def report(self):
    logging.info("Grain Dataset graph:\n\n%s", self._visualize_dataset_graph())


class _ExecutionStats(_VisualizationStats):
  """Execution time statistics for transformations."""

  def __init__(self, name: str, parents: Sequence[Stats]):
    super().__init__(name, parents)
    # Note that the buffer is intentionally not guarded by a lock to avoid lock
    # contention. Thread-safe operations are expected to only do atomic actions
    # on the buffer (such as `append`) making it safe due to GIL. See details in
    # https://docs.python.org/3/faq/library.html#what-kinds-of-global-value-mutation-are-thread-safe
    # The buffer is popped from a single(!) background reporting thread.
    self._self_times_buffer = []
    self._reporting_thread = None
    self._logging_thread = None
    self._reporting_thread_init_lock = threading.Lock()
    self._logging_thread_init_lock = threading.Lock()
    self._summary = execution_summary_pb2.ExecutionSummary.Node(
        min_processing_time_ns=sys.maxsize,
        max_processing_time_ns=0,
    )

  def __reduce__(self):
    return _ExecutionStats, (self.name, self._parents)

  def _reporting_loop(self):
    while True:
      time.sleep(_REPORTING_PERIOD_SEC)
      self.report()

  def _logging_execution_summary_loop(self):
    while True:
      time.sleep(_LOG_EXECTION_SUMMARY_PERIOD_SEC)
      summary = self._get_execution_summary()
      logging.info(
          "Grain Dataset Execution Summary:\n\n%s",
          _pretty_format_summary(summary),
      )

  def _build_execution_summary(
      self,
      execution_summary: execution_summary_pb2.ExecutionSummary,
      node_id: int,
  ):
    """Computes the stats summary for the whole dataset pipeline."""
    # By this point, all the nodes in the pipeline have been visited and
    # `_output_spec` & `name` has been set.
    self._summary.id = node_id
    self._summary.name = self.name
    self._summary.output_spec = str(self._output_spec)
    self._summary.is_output = self._is_output
    execution_summary.nodes.get_or_create(node_id)
    execution_summary.nodes[node_id].CopyFrom(self._summary)
    current_node_id = node_id
    for p in self._parents:
      node_id += 1
      execution_summary.nodes[current_node_id].inputs.append(node_id)
      # pytype: disable=attribute-error
      _, node_id = p._build_execution_summary(execution_summary, node_id)  # pylint: disable=protected-access
      # pytype: enable=attribute-error
    return execution_summary, node_id

  def _get_execution_summary(self) -> execution_summary_pb2.ExecutionSummary:
    """Returns ExecutionStats Summary for the dataset pipeline."""
    execution_summary = execution_summary_pb2.ExecutionSummary()
    result, _ = self._build_execution_summary(execution_summary, 0)
    return result

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0, num_produced_elements=1):
    start_time = time.perf_counter_ns()
    try:
      yield
    finally:
      self._self_times_buffer.append(
          time.perf_counter_ns() - start_time + offset_ns
      )
      if self._is_output:
        # We avoid acquiring `_reporting_thread_init_lock` here to avoid lock
        # contention.
        if self._reporting_thread is None:
          with self._reporting_thread_init_lock:
            # Check above together with update would not be atomic -- another
            # thread may have started the reporting thread.
            if self._reporting_thread is None:
              self._reporting_thread = threading.Thread(
                  target=self._reporting_loop, daemon=True
              )
              self._reporting_thread.start()

        if self._logging_thread is None:
          with self._logging_thread_init_lock:
            if self._logging_thread is None:
              self._logging_thread = threading.Thread(
                  target=self._logging_execution_summary_loop, daemon=True
              )
              self._logging_thread.start()

  def report(self):
    while self._self_times_buffer:
      # Each record in _self_times_buffer corresponds to a single element.
      self._summary.num_produced_elements += 1
      # Execution Summary must be cummulative from the beginning.
      self_time_ns = self._self_times_buffer.pop()
      self._summary.min_processing_time_ns = min(
          self._summary.min_processing_time_ns, self_time_ns
      )
      self._summary.max_processing_time_ns = max(
          self._summary.max_processing_time_ns, self_time_ns
      )
      self._summary.total_processing_time_ns += self_time_ns
      _self_time_ms_histogram.Record(self_time_ns, self.name)
    for p in self._parents:
      p.report()


def make_stats(name: str, parents: Sequence[Stats]) -> Stats:
  """Produces statistics instance according to the current execution mode."""
  return _NoopStats(name, parents=parents)
