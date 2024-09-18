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
import threading
import time
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

_EDGE_TEMPLATE = r"""{input_spec}
  ││
  ││  {transform}
  ││
  ╲╱
{output_spec}
"""


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
    self._accumulator = 0.0
    self._last = 0.0

  def __enter__(self):
    self._last = time.perf_counter()

  def __exit__(self, *args):
    self._accumulator += time.perf_counter() - self._last

  def value(self):
    """Returns the accumulated timer value across multiple usages."""
    return self._accumulator

  def reset(self):
    """Resets the accumulated timer value to 0."""
    self._accumulator = 0.0
    self._last = 0.0


class Stats(abc.ABC):
  """Base abstract class for statistics recording.

  This class replicates the transformation tree structure and provides
  interfaces for recording statistics in the given transformation node.
  """

  def __init__(self, name: str, parents: Sequence[Stats]):
    self._name = name
    self._output_spec = None
    self._parents = parents
    # Mark parent nodes as non-outputs. Nodes that are not updated are the
    # output nodes.
    self._is_output = True
    for p in parents:
      p._is_output = False

  @contextlib.contextmanager
  @abc.abstractmethod
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
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
      offset_sec: (Optional.) A offset to add to the self time measured by this
        function. Default to 0.0.
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

  def _log_dataset_graph(self):
    logging.info(
        "Generated dataset graph is:\n \n%s", self._visualize_dataset_graph()
    )

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
      transform_repr = self._name
    else:
      parent_vis = self._name
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
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass


class _VisualizationStats(Stats):
  """Produces Dataset Visualization Graph."""

  def __init__(self, name: str, parents: Sequence[Stats]):
    super().__init__(name, parents)
    self._graph_vis_thread = None
    self._graph_vis_thread_init_lock = threading.Lock()

  def __reduce__(self):
    return _VisualizationStats, (self._name, self._parents)

  @contextlib.contextmanager
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    yield

  def record_output_spec(self, element: T) -> T:
    # Visualize the dataset graph once last node had seen a non-None element
    if self._output_spec and self._is_output:
      if self._graph_vis_thread is None:
        with self._graph_vis_thread_init_lock:
          # The check above with update without a lock is not atomic,
          # need to check again under a lock.
          if self._graph_vis_thread is None:
            self._graph_vis_thread = threading.Thread(
                target=self._log_dataset_graph, daemon=True
            )
            self._graph_vis_thread.start()
    else:
      self._output_spec = tree.spec_like(element)
    return element

  def report(self):
    pass


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
    self._reporting_thread_init_lock = threading.Lock()

  def __reduce__(self):
    return _ExecutionStats, (self._name, self._parents)

  def _reporting_loop(self):
    while True:
      time.sleep(_REPORTING_PERIOD_SEC)
      self.report()

  @contextlib.contextmanager
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    start_time = time.perf_counter()
    try:
      yield
    finally:
      self._self_times_buffer.append(
          time.perf_counter() - start_time + offset_sec
      )
      # We avoid acquiring `_reporting_thread_init_lock` here to avoid lock
      # contention.
      if self._is_output and self._reporting_thread is None:
        with self._reporting_thread_init_lock:
          # Check above together with update would not be atomic -- another
          # thread may have started the reporting thread.
          if self._reporting_thread is None:
            self._reporting_thread = threading.Thread(
                target=self._reporting_loop, daemon=True
            )
            self._reporting_thread.start()

  def report(self):
    while self._self_times_buffer:
      _self_time_ms_histogram.Record(self._self_times_buffer.pop(), self._name)
    for p in self._parents:
      p.report()


def make_stats(name: str, parents: Sequence[Stats]) -> Stats:
  """Produces statistics instance according to the current execution mode."""
  return _NoopStats(name=name, parents=parents)
