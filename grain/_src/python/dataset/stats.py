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
import dataclasses
import os
import pprint
import sys
import threading
import time
import types
from typing import Any, Sequence, TypeVar

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

_MAX_COLUMN_WIDTH = 30
_MAX_ROW_LINES = 5

# We use a fixed-size length encoding for sending variable-length messages.
_ENCODED_LENGTH_SIZE = 10

# Maximum size of the buffer for large reads from fifo.
_READ_BUFFER_SIZE = 1024000


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
  return table._get_pretty_wrapped_summary()  # pylint: disable=protected-access


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

    self.contents = contents
    self._max_col_width = _MAX_COLUMN_WIDTH
    self.col_delim = col_delim
    self.col_widths = col_widths
    self._pretty_summary = []
    self.col_header = []

    # Determine the number of row_delim characters to fill the space used by
    # col_delim characters in a column header.
    col_delim_space_fill = len(self.col_delim) * (len(self.contents[0]) - 1)

    self.col_header.append(self.col_delim)
    for col_width in self.col_widths:
      if col_width > self._max_col_width:
        col_width = self._max_col_width
      self.col_header.append(row_delim * (col_width + 2))
    self.col_header.append(row_delim * (col_delim_space_fill))
    self.col_header.append(self.col_delim + "\n")
    self._pretty_summary.extend(self.col_header)

  def _get_pretty_wrapped_summary(self):
    """Wraps the table contents within the max column width and max row lines."""

    for row in self.contents:
      max_wrap = (max([len(i) for i in row]) // self._max_col_width) + 1
      max_wrap = min(max_wrap, _MAX_ROW_LINES)
      for r in range(max_wrap):
        self._pretty_summary.append(self.col_delim)
        for index in range(len(row)):
          if self.col_widths[index] > self._max_col_width:
            wrap = self._max_col_width
          else:
            wrap = self.col_widths[index]
          start = r * self._max_col_width
          end = (r + 1) * self._max_col_width
          self._pretty_summary.append(" ")
          self._pretty_summary.append(row[index][start:end].ljust(wrap))
          self._pretty_summary.append(" ")
          self._pretty_summary.append(self.col_delim)
        self._pretty_summary.append("\n")
      self._pretty_summary.extend(self.col_header)

    return "".join(self._pretty_summary)


def _merge_execution_summaries(
    aggregated_summary: execution_summary_pb2.ExecutionSummary,
    summary_from_worker: execution_summary_pb2.ExecutionSummary,
):
  """Merges the execution summary from the worker into the aggregated summary."""
  # we cannot use MergeFrom here because singular fields like
  # `max_processing_time_ns` will be overriden.
  for node_id in summary_from_worker.nodes:
    aggregated_summary.nodes[node_id].id = summary_from_worker.nodes[node_id].id
    aggregated_summary.nodes[node_id].name = summary_from_worker.nodes[
        node_id
    ].name
    aggregated_summary.nodes[node_id].output_spec = summary_from_worker.nodes[
        node_id
    ].output_spec
    aggregated_summary.nodes[node_id].is_output = summary_from_worker.nodes[
        node_id
    ].is_output
    aggregated_summary.nodes[node_id].ClearField("inputs")
    aggregated_summary.nodes[node_id].inputs.extend(
        summary_from_worker.nodes[node_id].inputs
    )
    if aggregated_summary.nodes[node_id].min_processing_time_ns == 0:
      aggregated_summary.nodes[node_id].min_processing_time_ns = (
          summary_from_worker.nodes[node_id].min_processing_time_ns
      )
    else:
      aggregated_summary.nodes[node_id].min_processing_time_ns = min(
          aggregated_summary.nodes[node_id].min_processing_time_ns,
          summary_from_worker.nodes[node_id].min_processing_time_ns,
      )
    aggregated_summary.nodes[node_id].max_processing_time_ns = max(
        aggregated_summary.nodes[node_id].max_processing_time_ns,
        summary_from_worker.nodes[node_id].max_processing_time_ns,
    )
    aggregated_summary.nodes[
        node_id
    ].total_processing_time_ns += summary_from_worker.nodes[
        node_id
    ].total_processing_time_ns
    aggregated_summary.nodes[
        node_id
    ].num_produced_elements += summary_from_worker.nodes[
        node_id
    ].num_produced_elements
  return aggregated_summary


def _update_execution_summary_in_main_process(
    summary_in_main_process: execution_summary_pb2.ExecutionSummary,
    summary_from_workers: execution_summary_pb2.ExecutionSummary,
):
  """Updates the execution summary in the main process by merging the summary from the workers."""
  num_nodes_in_main_process = len(summary_in_main_process.nodes)
  # The pipeline's output node within the workers becomes the input for the root
  # node in the main process. Therefore, the node IDs (and thus input IDs) in
  # the worker summary should be updated before merging it with the main
  # process's summary.

  for node_id in summary_from_workers.nodes:
    updated_node_id = node_id + num_nodes_in_main_process
    summary_from_workers.nodes[node_id].id = updated_node_id
    current_input_ids = summary_from_workers.nodes[node_id].inputs
    summary_from_workers.nodes[node_id].ClearField("inputs")
    for input_id in current_input_ids:
      summary_from_workers.nodes[node_id].inputs.append(
          input_id + num_nodes_in_main_process
      )
  input_ids = []
  root_node_in_main = None
  # Find the root node in the main process to update its inputs.
  for node_id in summary_in_main_process.nodes:
    if not getattr(summary_in_main_process.nodes[node_id], "inputs"):
      root_node_in_main = node_id
  for node_id in summary_from_workers.nodes:
    # If the node is an output node in the worker summary, it becomes the input
    # for the root node in the main process.
    if getattr(summary_from_workers.nodes[node_id], "is_output"):
      input_ids.append(summary_from_workers.nodes[node_id].id)
      summary_from_workers.nodes[node_id].is_output = False
    worker_node_id = summary_from_workers.nodes[node_id].id
    summary_in_main_process.nodes[worker_node_id].CopyFrom(
        summary_from_workers.nodes[node_id]
    )
  summary_in_main_process.nodes[root_node_in_main].inputs.extend(input_ids)
  return summary_in_main_process


class WorkerConnection:
  """A simplex connection for a worker to send data to the main process.

  Relies on already created fifo.

  Attributes:
    send_fifo: Path to a fifo for sending data. Must be opened for reading by
      the main process.
  """

  __slots__ = "send_fifo", "_send_fd"

  def __init__(self, send_fifo: str):
    self.send_fifo = send_fifo
    self._send_fd = -1

  def open(self) -> None:
    """Opens the connection.

    Blocks the caller until the main process opens the connection for read-only.
    """
    self._send_fd = os.open(self.send_fifo, os.O_WRONLY)  # pylint: disable=protected-access

  def send(self, data: bytes) -> None:
    """Sends data to the connection.

    Blocks the caller until the main process reads the data from the sending
    fifo.

    Args:
      data: bytes to send.
    """
    data_len = len(data)
    # Make fixed-size length encoding and send it over along with the data.
    encoded_len = f"{data_len:#0{_ENCODED_LENGTH_SIZE}x}".encode()
    os.write(self._send_fd, encoded_len)
    os.write(self._send_fd, data)


class MainConnection:
  """A simplex connection for the main process to receive data from workers.

  Relies on already created fifo.

  Attributes:
    recv_fifo: Path to a fifo for receiving data. Must be opened for writing by
      a worker.
  """

  __slots__ = "recv_fifo", "_recv_fd"

  def __init__(self, recv_fifo: str):
    self.recv_fifo = recv_fifo
    self._recv_fd = -1

  def open(self) -> None:
    """Opens the connection.

    Blocks the caller until the client end of the connection is opened.
    """
    self._recv_fd = os.open(self.recv_fifo, os.O_RDONLY)  # pylint: disable=protected-access

  def recv(self) -> bytes:
    """Reads data from the connection.

    Blocks the caller until the worker end sends data through the receiving
    fifo.

    Returns:
      bytes read from the connection.
    """
    # Reading the fixed number of leading bytes containing hex-encoded message
    # length.
    input_length = os.read(self._recv_fd, _ENCODED_LENGTH_SIZE)
    if not input_length:
      # Reached EOF. This means that the client end of the pipe was closed.
      return b""
    # Decode the length and read the necessary number of bytes.
    input_length = int(input_length, 16)
    input_parts = []
    read_length = 0
    while read_length < input_length:
      read_buffer_size = min(_READ_BUFFER_SIZE, input_length - read_length)
      buffer = os.read(self._recv_fd, read_buffer_size)
      input_parts.append(buffer)
      read_length += len(buffer)
    return b"".join(input_parts)


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


@dataclasses.dataclass(slots=True, kw_only=True)
class StatsConfig:
  """Statistics recording condiguration."""

  # Name of the current statistics recording node -- this is usually the name
  # of the current transformation.
  name: str
  # Whether this transformation mutates the element spec. This is used to
  # determine element spec of the current transformation.
  transform_mutates_spec: bool = True
  main_stats_connections: tuple[MainConnection, ...] | None = None
  worker_stats_fifo: str | None = None
  send_stats_to_main_process: bool = False


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

  def __reduce__(self):
    return _ExecutionStats, (self._config, self._parents)

  def _reporting_loop(self):
    while True:
      time.sleep(_REPORTING_PERIOD_SEC)
      self.report()

  def _send_summary_to_main_process_loop(self):
    """Sends the execution summary to the main process periodically."""
    connection = WorkerConnection(self._config.worker_stats_fifo)
    connection.open()
    while True:
      time.sleep(_LOG_EXECTION_SUMMARY_PERIOD_SEC)
      try:
        # Blocks until the main process reads the data.
        connection.send(self._get_execution_summary().SerializeToString())
      except BrokenPipeError:
        break

  def _get_combined_summary_from_workers(
      self,
  ) -> execution_summary_pb2.ExecutionSummary:
    """Returns the aggregated execution summary from all workers."""
    aggregated_summary_from_workers = execution_summary_pb2.ExecutionSummary()
    if self._config.main_stats_connections is not None:
      for connection in self._config.main_stats_connections:
        try:
          summary_from_worker = connection.recv()
          if not summary_from_worker:
            break
          summary_from_worker = (
              execution_summary_pb2.ExecutionSummary.FromString(
                  summary_from_worker
              )
          )
          # Combine the summary from all workers into a single summary.
          aggregated_summary_from_workers = _merge_execution_summaries(
              aggregated_summary_from_workers, summary_from_worker
          )
        except (BrokenPipeError, EOFError):
          break
        except Exception as e:  # pylint: disable=broad-except
          logging.exception("Failed to deserialize summary from worker: %s", e)
          break
    return aggregated_summary_from_workers

  def _get_aggregated_execution_summary(
      self,
  ) -> execution_summary_pb2.ExecutionSummary:
    """Returns the aggregated execution summary from all workers."""
    summary_in_main_process = self._get_execution_summary()
    summary_from_workers = self._get_combined_summary_from_workers()
    # Update the nodes in the main process with the aggregated summary from
    # all workers.
    aggregated_summary = _update_execution_summary_in_main_process(
        summary_in_main_process, summary_from_workers
    )
    return aggregated_summary

  def _logging_execution_summary_loop(self):
    """Logs the aggregated execution summary to the main process periodically."""
    while True:
      time.sleep(_LOG_EXECTION_SUMMARY_PERIOD_SEC)
      logging.info(
          "Grain Dataset Execution Summary:\n\n%s",
          _pretty_format_summary(self._get_aggregated_execution_summary()),
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
              if self._config.send_stats_to_main_process:
                self._logging_thread = threading.Thread(
                    target=self._send_summary_to_main_process_loop, daemon=True
                )
              else:
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


def make_stats(config: StatsConfig, parents: Sequence[Stats]) -> Stats:
  """Produces statistics instance according to the current execution mode."""
  return _NoopStats(config, parents=parents)
