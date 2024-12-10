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
from collections.abc import Sequence
import contextlib
import dataclasses
import enum
import pprint
import sys
import threading
import time
import types
from typing import Any, TypeVar

from absl import logging
from grain._src.core import config as grain_config
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
_LOG_EXECUTION_SUMMARY_PERIOD_SEC = 60
# Stop reporting if there has been no statistics updates for this long.
_REPORTING_TIMEOUT_SEC = 120

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

_MAX_COLUMN_WIDTH = 30
_MAX_ROW_LINES = 5


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
  # Remove the columns `output_spec` and `is_output` as they are available in
  # the visualization graph.
  col_names.remove("output_spec")
  col_names.remove("is_output")
  # Insert the average processing time column after the max processing time
  # column.
  index = col_names.index("max_processing_time_ns")
  col_names.insert(index + 1, _AVG_PROCESSING_TIME_COLUMN_NAME)

  tabular_summary.append(
      [_COLUMN_NAME_OVERRIDES.get(name, name) for name in col_names]
  )

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

  for node_id in sorted(summary.nodes, reverse=True):
    row_values = []
    for name in col_names:
      is_total_processing_time_zero = (
          summary.nodes[node_id].total_processing_time_ns == 0
      )
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
          col_value = "N/A"
        elif name != "num_produced_elements":
          col_value = _pretty_format_ns(value)
        else:
          col_value = str(value)
      else:
        col_value = str(value)
      row_values.append(col_value)
    tabular_summary.append(row_values)
  table = _Table(tabular_summary, col_widths=col_widths)
  return table.get_pretty_wrapped_summary()  # pylint: disable=protected-access


@enum.unique
class ExecutionTrackingMode(enum.Flag):
  """Represents different modes for tracking execution statistics.

  Available modes:
    DISABLED:
      No execution statistics are measured. This mode is the default.
    STAGE_TIMING:
      The time taken for each transformation stage to execute is measured and
      recorded. This recorded time reflects the duration spent within the
      specific transformation to return an element, excluding the time spent in
      any parent transformations. The recorded time can be retrieved using
      `grain.experimental.get_execution_summary` method.
  """

  DISABLED = enum.auto()
  STAGE_TIMING = enum.auto()


class _Table:
  """Table class for pretty printing tabular data."""

  def __init__(
      self,
      contents,
      *,
      col_widths,
      col_delim="|",
      row_delim="-",
  ):

    self._contents = contents
    self._max_col_width = _MAX_COLUMN_WIDTH
    self._col_delim = col_delim
    self._col_widths = col_widths
    self._pretty_summary = []
    self._col_header = []

    # Determine the number of row_delim characters to fill the space used by
    # col_delim characters in a column header.
    col_delim_space_fill = len(self._col_delim) * (len(self._contents[0]) - 1)

    self._col_header.append(self._col_delim)
    for col_width in self._col_widths:
      if col_width > self._max_col_width:
        col_width = self._max_col_width
      self._col_header.append(row_delim * (col_width + 2))
    self._col_header.append(row_delim * (col_delim_space_fill))
    self._col_header.append(self._col_delim + "\n")
    self._pretty_summary.extend(self._col_header)

  def get_pretty_wrapped_summary(self):
    """Wraps the table contents within the max column width and max row lines."""

    for row in self._contents:
      max_wrap = (max([len(i) for i in row]) // self._max_col_width) + 1
      max_wrap = min(max_wrap, _MAX_ROW_LINES)
      for r in range(max_wrap):
        self._pretty_summary.append(self._col_delim)
        for index in range(len(row)):
          if self._col_widths[index] > self._max_col_width:
            wrap = self._max_col_width
          else:
            wrap = self._col_widths[index]
          start = r * self._max_col_width
          end = (r + 1) * self._max_col_width
          self._pretty_summary.append(" ")
          self._pretty_summary.append(row[index][start:end].ljust(wrap))
          self._pretty_summary.append(" ")
          self._pretty_summary.append(self._col_delim)
        self._pretty_summary.append("\n")
      self._pretty_summary.extend(self._col_header)

    return "".join(self._pretty_summary)


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


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StatsConfig:
  """Statistics recording condiguration."""

  # Name of the current statistics recording node -- this is usually the name
  # of the current transformation.
  name: str
  # Whether this transformation mutates the element spec. This is used to
  # determine element spec of the current transformation.
  transform_mutates_spec: bool = True
  # Whether to log the execution summary.
  log_summary: bool = False


class Stats(abc.ABC):
  """Base abstract class for statistics recording.

  This class replicates the transformation tree structure and provides
  interfaces for recording statistics in the given transformation node.
  """

  def __init__(self, config: StatsConfig, parents: Sequence[Stats]):
    self._config = config
    self._self_output_spec = None
    self._parents = parents
    # Mark parent nodes as non-outputs. Nodes that are not updated are the
    # output nodes.
    self._is_output = True
    for p in parents:
      p._is_output = False

  @property
  def output_spec(self) -> Any:
    if self._config.transform_mutates_spec:
      return self._self_output_spec
    assert self._parents
    return self._parents[0].output_spec

  @contextlib.contextmanager
  @abc.abstractmethod
  def record_self_time(self, offset_ns: float = 0):
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
      transform_repr = self._config.name
    else:
      parent_vis = self._config.name
      transform_repr = ""
    return (
        has_multiple_parents,
        _EDGE_TEMPLATE.format(
            input_spec=parent_vis,
            transform=transform_repr,
            output_spec=pprint.pformat(self.output_spec),
        ),
    )


class _NoopStats(Stats):
  """Default implementation for statistics collection that does nothing."""

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass


class _VisualizationStats(Stats):
  """Produces Dataset Visualization Graph."""

  def __init__(self, config: StatsConfig, parents: Sequence[Stats]):
    super().__init__(config, parents)
    self._reported = False
    self._reported_lock = threading.Lock()

  def __reduce__(self):
    return _VisualizationStats, (self._config.name, self._parents)

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0):
    yield

  def record_output_spec(self, element: T) -> T:
    # Visualize the dataset graph once last node had seen a non-None element.
    if self._self_output_spec is None:
      self._self_output_spec = tree.spec_like(element)
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

  def __init__(self, config: StatsConfig, parents: Sequence[Stats]):
    super().__init__(config, parents)
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
    self._last_update_time = 0
    self._last_report_time = 0

  def __reduce__(self):
    return _ExecutionStats, (self._config, self._parents)

  def _should_report(self):
    return time.time() - self._last_update_time < _REPORTING_TIMEOUT_SEC

  def _reporting_loop(self):
    while self._should_report():
      time.sleep(_REPORTING_PERIOD_SEC)
      # A node can be marked as non-output after the corresponding
      # transformation started processing elements -- we do not control the
      # initialization time.
      if not self._is_output:
        return
      self.report()

  def _logging_execution_summary_loop(self):
    while self._should_report():
      time.sleep(_LOG_EXECUTION_SUMMARY_PERIOD_SEC)
      # A node can be marked as non-output after the corresponding
      # transformation started processing elements -- we do not control the
      # initialization time.
      if not self._is_output:
        return
      if self._last_update_time > self._last_report_time:
        self._last_report_time = time.time()
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
    self._summary.name = self._config.name
    self._summary.output_spec = str(self.output_spec)
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
  def record_self_time(self, offset_ns: int = 0):
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
        self._last_update_time = time.time()
        if self._reporting_thread is None:
          with self._reporting_thread_init_lock:
            # Check above together with update would not be atomic -- another
            # thread may have started the reporting thread.
            if self._reporting_thread is None:
              self._reporting_thread = threading.Thread(
                  target=self._reporting_loop, daemon=True
              )
              self._reporting_thread.start()
        if self._config.log_summary and self._logging_thread is None:
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
      _self_time_ms_histogram.Record(self_time_ns, self._config.name)
    for p in self._parents:
      p.report()


def make_stats(
    config: StatsConfig,
    parents: Sequence[Stats],
    execution_tracking_mode: ExecutionTrackingMode = ExecutionTrackingMode.DISABLED,
) -> Stats:
  """Produces statistics instance according to the current execution mode."""
  return _NoopStats(config, parents=parents)
