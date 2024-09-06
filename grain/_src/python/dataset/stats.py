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
import threading
import time
from typing import Callable, Sequence, TypeVar

from absl import logging
from grain._src.core import monitoring as grain_monitoring

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
# Timeout before a lock can be acquired to collect statistics for each stats
# recording
_LOCK_ACQUISITION_TIMEOUT_SEC = 0.001  # 1 msec
# Time between two consecutive monitoring reporting.
_MONITORING_PERIOD_SEC = 20


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
    self._parents = parents
    # Mark parent nodes as non-outputs. Nodes that are not updated are the
    # output nodes.
    self._is_output = True
    for p in parents:
      p._is_output = False
    self._lock = threading.Lock()

  def __getstate__(self):
    state = self.__dict__.copy()
    del state["_lock"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._lock = threading.Lock()

  @contextlib.contextmanager
  @abc.abstractmethod
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    """Records time spent in this node's transfromation.

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
    """
    ...

  def _for_each_parent(self, fn: Callable[[Stats], None], visited: set[Stats]):
    if self in visited:
      return
    fn(self)
    visited.add(self)
    for p in self._parents:
      p._for_each_parent(fn, visited)  # pylint: disable=protected-access


class NoopStats(Stats):
  """Default implementation for statistics collection that does nothing."""

  @contextlib.contextmanager
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass


class ExecutionStats(Stats):
  """Execution time statistics for transformations."""

  def __init__(self, name: str, parents: Sequence[Stats]):
    super(ExecutionStats, self).__init__(name, parents)
    self._thread = None
    self._monitoring_period_sec = _MONITORING_PERIOD_SEC
    self._lock_timeout_sec = _LOCK_ACQUISITION_TIMEOUT_SEC
    self._num_elements = 0
    self._self_time_sec = 0.0

  def _report_monitoring_thread(self):

    def report_monitoring(stats: Stats) -> None:
      stats.report()

    # Reports monitoring and goes to sleep for the requestion duration.
    while True:
      visited = set()
      self._for_each_parent(report_monitoring, visited)
      time.sleep(self._monitoring_period_sec)

  @contextlib.contextmanager
  def record_self_time(self, offset_sec: float = 0.0, num_produced_elements=1):
    start_time = time.perf_counter()
    try:
      yield
    finally:
      if self._monitoring_period_sec >= 0:
        # Only record the self time if the lock can be acquired in a short time.
        if self._lock.acquire(timeout=self._lock_timeout_sec):
          self._self_time_sec += time.perf_counter() - start_time + offset_sec
          self._num_elements += num_produced_elements
          self._lock.release()
          # A separate thread is used to monitor the execution time of the whole
          # pipeline. This thread is only started for the outputs of the
          # pipeline
          if self._is_output and self._thread is None:
            logging.info("Starting monitoring thread for %s.", self._name)
            self._thread = threading.Thread(
                target=self._report_monitoring_thread, daemon=True
            )
            self._thread.start()

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    self_time_ms = None
    with self._lock:
      if self._num_elements > 0:
        self_time_ms = int(self._self_time_sec / self._num_elements * 1000.0)
        self._self_time_sec = 0.0
        self._num_elements = 0
    if self_time_ms is not None:
      _self_time_ms_histogram.Record(self_time_ms, self._name)
