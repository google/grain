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
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import functools
from multiprocessing import queues
import pprint
import queue
import sys
import threading
import time
import types
from typing import Any, TypeVar
import weakref

from absl import logging
from grain._src.core import config as grain_config
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import tree_lib
from grain._src.python.dataset import base
from grain._src.python.dataset import stats_utils
from grain.proto import execution_summary_pb2

from grain._src.core import monitoring


# Registry of weak references to output dataset iterators for collecting
# execution stats.
_iter_weakref_registry = []

_self_time_ns_histogram = monitoring.EventMetric(
    "/grain/python/dataset/self_time_ns",
    metadata=monitoring.Metadata(
        description=(
            "Histogram of transformation self time. Each data point is the "
            "average value of self times/element produced during a monitoring "
            "interval."
        ),
        units=monitoring.Units.NANOSECONDS,
    ),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
    bucketer=monitoring.Bucketer.PowersOf(2.0),
)

_next_duration_ns_histogram = monitoring.EventMetric(
    "/grain/python/dataset/next_duration_ns",
    metadata=monitoring.Metadata(
        description=(
            "Histogram of durations of every `__next__` call on the output"
            " iterator. Each data point is the duration value of `__next__`"
            " call."
        ),
        units=monitoring.Units.NANOSECONDS,
    ),
    root=grain_monitoring.get_monitoring_root(),
    bucketer=monitoring.Bucketer.PowersOf(2.0),
)

T = TypeVar("T")
# Time between two consecutive monitoring reports.
_REPORTING_PERIOD_SEC = 10
_LOG_EXECUTION_SUMMARY_PERIOD_SEC = 60
_WORKER_QUEUE_TIMEOUT_SEC = 2
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
_MEMORY_USAGE_COLUMN_NAME = "memory usage"

_COLUMN_NAME_OVERRIDES = types.MappingProxyType({
    "wait_time_ratio": "percent wait time",
    "min_processing_time_ns": "min processing time",
    "max_processing_time_ns": "max processing time",
    "total_processing_time_ns": "total processing time",
    "num_produced_elements": "num produced elements",
    "output_spec": "output spec",
})

_PROCESSING_TIME_COLUMNS = (
    "min_processing_time_ns",
    "max_processing_time_ns",
    "total_processing_time_ns",
    _AVG_PROCESSING_TIME_COLUMN_NAME,
)

_MAX_COLUMN_WIDTH = 30
_MAX_ROW_LINES = 5


def _get_nodes_before_prefetch(
    node: int, summary: execution_summary_pb2.ExecutionSummary
) -> list[int]:
  """Returns nodes in the path from a given node to a prefetch node."""
  child_nodes = []
  nodes_to_visit = [node]
  while nodes_to_visit:
    node_id = nodes_to_visit.pop()
    node = summary.nodes[node_id]
    child_nodes.append(node_id)
    if node.is_prefetch:
      continue  # Skip adding inputs for the prefetch node
    nodes_to_visit.extend(node.inputs)
  return child_nodes


def _find_aggregated_processing_time(
    summary: execution_summary_pb2.ExecutionSummary,
    node_ids: list[int],
) -> int:
  """Finds aggregated processing time for the given node IDs."""
  return sum(
      summary.nodes[node_id].total_processing_time_ns for node_id in node_ids
  )


def _compute_wait_time_ratio(
    summary: execution_summary_pb2.ExecutionSummary,
    node_id: int,
    aggregated_wait_time_ns: int,
    prefetch_factor: int = 1,
) -> None:
  """Computes the wait time ratio for all the nodes in the execution summary.

  Args:
    summary: The execution summary to update the `wait_time_ratio` for.
    node_id: The current node for which to compute the `wait_time_ratio`.
    aggregated_wait_time_ns: The aggregated wait time of the nodes running under
      prefetch.
    prefetch_factor: The factor by which to multiply the `total_processing_time`
      of the node to get it's wait time ratio.
  """
  if aggregated_wait_time_ns == 0:
    return
  node = summary.nodes[node_id]
  node_wait_ratio = prefetch_factor * (
      node.total_processing_time_ns / aggregated_wait_time_ns
  )
  node.wait_time_ratio = round(node_wait_ratio, 4)
  for input_node_id in node.inputs:
    # If the node is executed in multiple threads, the iterator's wait time
    # ratio attributed to the prefetch node is distributed among these nodes
    # proportionally to their total processing time.
    if node.is_prefetch:
      prefetch_factor = node.wait_time_ratio
      prefetch_child_nodes = _get_nodes_before_prefetch(input_node_id, summary)
      aggregated_wait_time_ns = _find_aggregated_processing_time(
          summary, prefetch_child_nodes
      )
      # The `wait_time_ratio` of the prefetch node is sum of `wait_time_ratio`
      # of all the nodes running under it. Here we set it to 0 as it is already
      # accounted for in the ancestor nodes and sum of `wait_time_ratio` of all
      # the nodes in a pipeline should be 1.
      node.wait_time_ratio = 0
    _compute_wait_time_ratio(
        summary,
        input_node_id,
        aggregated_wait_time_ns,
        prefetch_factor,
    )


def _populate_wait_time_ratio(
    summary: execution_summary_pb2.ExecutionSummary,
) -> None:
  """Populates the `wait_time_ratio` for all the nodes in the execution summary."""
  iterator_nodes = _get_nodes_before_prefetch(0, summary)
  aggregated_wait_time_ns = _find_aggregated_processing_time(
      summary, iterator_nodes
  )
  _compute_wait_time_ratio(summary, 0, aggregated_wait_time_ns)


def _pretty_format_col_value(
    value: Any,
    name: Any,
) -> str:
  """Pretty formats the column value for the given column name."""
  if name == "wait_time_ratio":
    value = stats_utils.format_ratio_as_percent(value)
  elif name == _MEMORY_USAGE_COLUMN_NAME:
    value = stats_utils.pretty_format_bytes(value)
  elif name in _PROCESSING_TIME_COLUMNS:
    value = stats_utils.pretty_format_ns(value)
  return str(value)


def _get_col_value(
    name: str, node: execution_summary_pb2.ExecutionSummary.Node
) -> str:
  """Returns the string representation of the value of a column for the given node."""
  if node.is_prefetch:
    # The prefetch's iterator wait time of the prefetch node is distributed
    # across its child nodes.
    if name == "wait_time_ratio":
      return "N/A"

  # If the total processing time is zero, the pipeline has not yet
  # produced an element and processing times & num_produced_elements are
  # not yet meaningful.
  if name in _PROCESSING_TIME_COLUMNS and node.total_processing_time_ns == 0:
    return "N/A"

  if name == _AVG_PROCESSING_TIME_COLUMN_NAME:
    value = stats_utils.get_avg_processing_time_ns(node)
  elif name == _MEMORY_USAGE_COLUMN_NAME:
    # Only calculate for prefetch nodes, otherwise default to 0.
    # In multiprocessing, stats reporting is asynchronous, so at times,
    # `bytes_produced` in main process might be more than the `bytes_consumed`
    # from child process.
    value = (
        max(0, node.bytes_consumed - node.bytes_produced)
        if node.is_prefetch
        else 0
    )
  else:
    value = getattr(node, name)
  return _pretty_format_col_value(value, name)


def _get_col_names(
    summary: execution_summary_pb2.ExecutionSummary,
) -> list[str]:
  """Returns all the column names for the given execution stats."""
  col_names = [key for key in summary.nodes[0].DESCRIPTOR.fields_by_name.keys()]
  # Remove the columns `output_spec` and `is_output` as they are available in
  # the visualization graph.
  col_names.remove("output_spec")
  col_names.remove("is_output")
  col_names.remove("is_prefetch")
  col_names.remove("bytes_consumed")
  col_names.remove("bytes_produced")
  # Insert the average processing time column after the max processing time
  # column.
  index = col_names.index("max_processing_time_ns")
  col_names.insert(index + 1, _AVG_PROCESSING_TIME_COLUMN_NAME)
  col_names.append(_MEMORY_USAGE_COLUMN_NAME)
  return col_names


def pretty_format_summary(
    summary: execution_summary_pb2.ExecutionSummary,
) -> str:
  """Returns Execution Stats Summary for the dataset pipeline in tabular format."""
  tabular_summary = []
  col_names = _get_col_names(summary)
  tabular_summary.append(
      [_COLUMN_NAME_OVERRIDES.get(name, name) for name in col_names]
  )

  # Compute the maximum width of each column.
  col_widths = []
  for name in col_names:
    max_width = len(_COLUMN_NAME_OVERRIDES.get(name, name))
    for node in summary.nodes.values():
      max_width = max(len(_get_col_value(name, node)), max_width)
    col_widths.append(max_width)

  for node_id in sorted(summary.nodes, reverse=True):
    node = summary.nodes[node_id]
    row_values = [_get_col_value(name, node) for name in col_names]
    tabular_summary.append(row_values)
  table = _Table(tabular_summary, col_widths=col_widths)
  return table.get_pretty_wrapped_summary()


def record_next_duration_if_output(next_fn):
  """Records the duration of the `__next__` call on the output iterator node.

  Expected to be used as follows:
  ```
  class MyMapDatasetIterator(DatasetIterator):
    ...
    @stats.record_next_duration_if_output
    def __next__(self):
      ...
  ```

  Args:
    next_fn: The `__next__` function to wrap.

  Returns:
    The wrapped `next_fn`.
  """

  @functools.wraps(next_fn)
  def wrapper(iterator):
    start_time = time.perf_counter_ns()
    result = next_fn(iterator)

    if iterator._stats._is_output:  # pylint:disable=protected-access
      next_duration_ns = time.perf_counter_ns() - start_time
      _next_duration_ns_histogram.Record(next_duration_ns)
    return result

  return wrapper


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


@dataclasses.dataclass(kw_only=True)
class StatsConfig:
  """Statistics recording condiguration."""

  # Name of the current statistics recording node -- this is usually the name
  # of the current transformation.
  name: str
  # Whether this transformation mutates the element spec. This is used to
  # determine element spec of the current transformation.
  transform_mutates_spec: bool = True
  # Whether this transformation is a prefetch transformation.
  is_prefetch: bool = False
  # Whether to log the execution summary.
  log_summary: bool = False
  # Queue for each child process from which to receive execution summaries in
  # the main process. This is only populated to the stats object of the
  # `mp_prefetch` transformation node.
  stats_in_queues: tuple[queues.Queue, ...] | None = None
  # Queue used by child processes to send it's execution summary to the main
  # process. This queue is only populated to the stats object of the output node
  # in the pipeline within the child process.
  stats_out_queue: queues.Queue | None = None
  # Weak reference to the iterator that this stats object is associated with.
  iter_weakref: weakref.ReferenceType | None = None

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["iter_weakref"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.iter_weakref = None


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
    _iter_weakref_registry.append(config.iter_weakref)
    for p in parents:
      p._is_output = False
      if p._config.iter_weakref in _iter_weakref_registry:
        _iter_weakref_registry.remove(p._config.iter_weakref)

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

  @abc.abstractmethod
  def record_bytes_consumed(self, element: Any) -> Any:
    """Records bytes consumed by this node."""
    ...

  @abc.abstractmethod
  def record_bytes_produced(self, element: Any) -> Any:
    """Records bytes produced by this node."""
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


def _running_in_colab() -> bool:
  """Returns whether the current process is running in Colab."""
  return "google.colab" in sys.modules


class _DefaultStats(Stats):
  """Default implementation for statistics collection that does nothing."""

  @contextlib.contextmanager
  def record_self_time(self, offset_ns: int = 0):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass

  def record_bytes_consumed(self, element: Any) -> Any:
    return element

  def record_bytes_produced(self, element: Any) -> Any:
    return element


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
      self._self_output_spec = tree_lib.spec_like(element)
      if self._is_output and not self._reported:
        # The check above with update without a lock is not atomic, need to
        # check again under a lock.
        with self._reported_lock:
          if not self._reported:
            self.report()
            self._reported = True
    return element

  def record_bytes_consumed(self, element: Any) -> Any:
    return element

  def record_bytes_produced(self, element: Any) -> Any:
    return element

  def report(self):
    msg = f"Grain Dataset graph:\n\n{self._visualize_dataset_graph()}"
    logging.info(msg)
    if _running_in_colab():
      print(msg)


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
    self._produced_memory_buffer = []
    self._consumed_memory_buffer = []
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

  def _send_stats_to_main_process_loop(self) -> None:
    """Puts the execution summary into the main process queue periodically."""
    stats_queue = self._config.stats_out_queue
    if stats_queue is None:
      raise ValueError(
          "Worker failed to report summary, `stats_out_queue` is None"
      )
    while self._should_report():
      time.sleep(_LOG_EXECUTION_SUMMARY_PERIOD_SEC)
      try:
        stats_queue.put(self._get_execution_summary())
      except queue.Full:
        logging.info(
            "Couldn't send execution summary from child process. Queue full"
        )
        pass

  def _logging_execution_summary_loop(self) -> None:
    """Logs the execution summary periodically."""
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
        summary = stats_utils.sort_nodes_by_wait_time_ratio(summary)
        msg = (
            "Grain Dataset Execution Summary:\n\nNOTE: Before analyzing the"
            " `total_processing_time` for a node, please check the `percent"
            " wait time` column to ensure that the node is indicated as"
            " bottleneck. The `MapDataset` nodes are executed in multiple"
            " threads and thus, should not be compared to the"
            " `total_processing_time` of `DatasetIterator` nodes."
            f"\n\n{pretty_format_summary(summary)}"
        )
        for warning in stats_utils.analyze_summary(summary):
          msg += "\n" + warning
        logging.info(msg)
        if _running_in_colab():
          print(msg)

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
    self._summary.is_prefetch = self._config.is_prefetch
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
    _populate_wait_time_ratio(result)
    return result

  def _flush_execution_summary_loop(self) -> Callable[[], None]:
    """Returns the loop function to be executed by the logging thread."""
    if self._config.stats_out_queue:
      return self._send_stats_to_main_process_loop
    else:
      return self._logging_execution_summary_loop

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
            # Check above together with update would not be atomic -- another
            # thread may have started the logging thread.
            if self._logging_thread is None:
              self._logging_thread = threading.Thread(
                  target=self._flush_execution_summary_loop(), daemon=True
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
      _self_time_ns_histogram.Record(self_time_ns, self._config.name)
    while self._consumed_memory_buffer:
      self._summary.bytes_consumed += self._consumed_memory_buffer.pop()
    while self._produced_memory_buffer:
      self._summary.bytes_produced += self._produced_memory_buffer.pop()
    for p in self._parents:
      p.report()

  def record_bytes_consumed(self, element: Any) -> Any:
    """Records the memory usage of the element."""
    return stats_utils.record_size(element, self._consumed_memory_buffer)

  def record_bytes_produced(self, element: Any) -> Any:
    """Records the memory usage of the element."""
    return stats_utils.record_size(element, self._produced_memory_buffer)


class _MPPrefetchExecutionStats(_ExecutionStats):
  """Execution time statistics for multiprocess prefetch transformations.

  This class is responsible for aggregating the execution summary from all
  workers and building the complete execution summary in the main process.
  """

  def __init__(self, config: StatsConfig, parents: Sequence[Stats]):
    super().__init__(config, parents)
    self._merged_summary_from_workers = execution_summary_pb2.ExecutionSummary()

  def _get_merged_summary_from_workers(
      self,
  ) -> execution_summary_pb2.ExecutionSummary:
    """Calculates the aggregated execution summary from all workers."""
    aggregated_summary_from_workers = execution_summary_pb2.ExecutionSummary()
    stats_in_queues = self._config.stats_in_queues
    for worker_index, worker_queue in enumerate(stats_in_queues):
      try:
        summary_from_worker = worker_queue.get(
            timeout=_WORKER_QUEUE_TIMEOUT_SEC
        )
        # Combine the summary from all workers into a single summary.
        aggregated_summary_from_workers = stats_utils.merge_execution_summaries(
            aggregated_summary_from_workers, summary_from_worker
        )
      except queue.Empty:
        logging.warning(
            "Couldn't get execution summary from the child process %i",
            worker_index,
        )
        return self._merged_summary_from_workers
    self._merged_summary_from_workers = aggregated_summary_from_workers
    return self._merged_summary_from_workers

  def _build_execution_summary(
      self,
      execution_summary: execution_summary_pb2.ExecutionSummary,
      node_id: int,
  ) -> tuple[execution_summary_pb2.ExecutionSummary, int]:
    # By this point, all the nodes in the pipeline have been visited and
    # `_output_spec` & `name` has been set.
    self._summary.id = node_id
    self._summary.name = self._config.name
    self._summary.output_spec = str(self.output_spec)
    self._summary.is_output = self._is_output
    self._summary.is_prefetch = self._config.is_prefetch
    execution_summary.nodes.get_or_create(node_id)
    execution_summary.nodes[node_id].CopyFrom(self._summary)
    current_node_id = node_id
    combined_summary_from_workers = execution_summary_pb2.ExecutionSummary()
    combined_summary_from_workers.CopyFrom(
        self._get_merged_summary_from_workers()
    )
    execution_summary = stats_utils.get_complete_summary(
        execution_summary, combined_summary_from_workers, current_node_id + 1
    )
    for p in self._parents:
      node_id += 1
      execution_summary.nodes[current_node_id].inputs.append(node_id)
      assert isinstance(p, _ExecutionStats)
      _, node_id = p._build_execution_summary(execution_summary, node_id)  # pylint: disable=protected-access
    return execution_summary, node_id


def make_stats(
    config: StatsConfig,
    parents: Sequence[Stats],
    execution_tracking_mode: base.ExecutionTrackingMode = (
        base.ExecutionTrackingMode.DISABLED
    ),
) -> Stats:
  """Produces statistics instance according to the current execution mode."""
  vis_output_dir = grain_config.config.get_or_default(
      "py_dataset_visualization_output_dir"
  )
  # Only None and "" are supported.
  if vis_output_dir:
    raise NotImplementedError(
        "Saving the dataset graph to a file is not supported yet. Set"
        " `grain_py_dataset_visualization_output_dir` to empty string to"
        " produce visualization in the logs."
    )
  if grain_config.config.get_or_default("py_debug_mode"):
    # In debug mode, we always log the execution summary.
    config = dataclasses.replace(config, log_summary=True)
    if config.stats_in_queues:
      return _MPPrefetchExecutionStats(config, parents)
    return _ExecutionStats(config, parents=parents)
  if execution_tracking_mode == base.ExecutionTrackingMode.STAGE_TIMING:
    return _ExecutionStats(config, parents=parents)
  if vis_output_dir is not None:
    return _VisualizationStats(config, parents=parents)
  return _DefaultStats(config, parents=parents)
