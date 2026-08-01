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
"""Implements LazyDataset elements prefetching."""

from __future__ import annotations

import collections
from collections.abc import Iterator, Sequence
import copy
import functools
from multiprocessing import queues
import queue
import sys
import threading
import typing
from typing import Any, Optional, Protocol, TypeVar

from absl import logging
from concurrent import futures
from grain._src.core import monitoring as grain_monitoring
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import filter as filter_dataset
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import source
from grain._src.python.ipc import variable_size_queue

_prefetch_ready_elements = grain_monitoring.EventMetric(
    "/grain/python/dataset/prefetch_buffer_ready_count",
    metadata=grain_monitoring.Metadata(
        description=(
            "Distribution of the number of consecutive elements from the"
            " front of the prefetch buffer that are ready for consumption."
            " Low values indicate a possible IO bottleneck."
        ),
    ),
)

T = TypeVar("T")


def _initialize_prefetch_stats(
    iterator: dataset.DatasetIterator[Any],
    execution_tracking_mode: base.ExecutionTrackingMode,
    parent_stats: Sequence[dataset_stats.Stats],
    stats_in_queues: Optional[tuple[queues.Queue[Any], ...]] = None,
) -> dataset_stats.Stats:
  """Helper to initialize stats for prefetch iterators."""
  config = dataset_stats.StatsConfig(
      name=str(iterator),
      transform_mutates_spec=iterator._MUTATES_ELEMENT_SPEC,  # pylint: disable=protected-access
      node_type=dataset_stats.NodeType.PREFETCH,
      iter_weakref=dataset_stats.HashableWeakRef(iterator),
  )
  if stats_in_queues is not None:
    config.stats_in_queues = stats_in_queues

  # If the stats object has already been initialized, copy the queues from
  # the original stats object to the new stats object.
  if "_stats" in iterator.__dict__:
    # pylint: disable=protected-access
    config.stats_out_queue = iterator._stats._config.stats_out_queue
    config.stats_in_queues = iterator._stats._config.stats_in_queues
    # pylint: enable=protected-access

  return dataset_stats.make_stats(
      config,
      parent_stats,
      execution_tracking_mode=execution_tracking_mode,
  )


@dataset_stats.trace_input_pipeline_prefetch
def _getitem(
    stats: dataset_stats.Stats, parent: dataset.MapDataset[T], index: int
) -> T:
  """Helper to record the memory usage of the element before prefetching."""
  return stats.record_bytes_consumed(parent[index])


class PrefetchIterDataset(dataset.IterDataset[T]):
  """Iterable dataset that uses a thread pool for prefetching."""

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      *,
      read_options: grain_options.ReadOptions,
      allow_nones: bool = False,
  ):
    super().__init__(parent)
    self._read_options = read_options
    self._allow_nones = allow_nones

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    """Replaces `MapDataset` parents with their sliced versions."""
    assert isinstance(self._parent, dataset.MapDataset), self._parent
    if not sequential_slice:
      self._parents = (self._parent.slice(sl),)
    else:
      dataset.set_slice(self._parent, sl, sequential_slice)

  def __str__(self) -> str:
    return (
        f"PrefetchIterDataset(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return PrefetchDatasetIterator(
        self._parent, self._read_options, self._allow_nones  # pyrefly: ignore[bad-argument-type]
    )

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


class PrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a thread pool."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      read_options: grain_options.ReadOptions,
      allow_nones: bool,
  ):
    # Note that the parent is not a conventional iterator, but a MapDataset.
    super().__init__()
    self._map_parent = parent
    self._dataset_length = len(parent)
    self._read_options = read_options
    self._next_returned_index = 0
    self._next_buffered_index = 0
    self._buffer = collections.deque()
    self._lock = threading.Lock()
    self._executor_wrapper = None

    assert isinstance(read_options.num_threads, int)
    assert isinstance(read_options.prefetch_buffer_size, int)
    self._target_num_threads = read_options.num_threads
    self._target_prefetch_buffer_size = read_options.prefetch_buffer_size

    self._allow_nones = allow_nones
    if self._target_prefetch_buffer_size > 0 and self._target_num_threads > 0:
      self._executor = futures.ThreadPoolExecutor(
          self._target_num_threads, thread_name_prefix="grain-prefetch"
      )

  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ):
    parent_stats = self._map_parent._initialize_stats(execution_tracking_mode)  # pylint: disable=protected-access
    # Connect to `MapDataset` parent stats.
    self._stats = _initialize_prefetch_stats(
        self, execution_tracking_mode, (parent_stats,)
    )
    return self._stats

  @functools.cached_property
  def _stats(self):
    return self._initialize_stats(
        self._ctx.dataset_options.execution_tracking_mode  # pyrefly: ignore[bad-argument-type]
    )

  @functools.cached_property
  def _threshold_checker(self):
    # Sparse `MapDataset` transformations produce Nones which we filter out
    # here. The validator helps to detect if we discard too many elements.
    return filter_dataset.FilterThresholdChecker(
        transform_name=str(self),
        warn_threshold=self._ctx.dataset_options.filter_warn_threshold_ratio,  # pyrefly: ignore[bad-argument-type]
        raise_threshold=self._ctx.dataset_options.filter_raise_threshold_ratio,  # pyrefly: ignore[bad-argument-type]
    )

  def _measure_prefetch_depth(self):
    ready_count = 0
    for future in self._buffer:
      if future.done():
        ready_count += 1
      else:
        break
    _prefetch_ready_elements.Record(ready_count)

  @dataset_stats.record_next_duration_if_output
  @dataset_stats.trace_input_pipeline_next(
      stage_category=dataset_stats.IPL_CAT_PREFETCH
  )
  def __next__(self) -> T:
    self._assert_not_closed()
    # The time recorded here is the time spent in prefetch node to return an
    # element, including the time spent in parent node.
    timer = dataset_stats.Timer()
    # We loop here to skip all None elements (in case the underlying dataset
    # is sparse), if self._allow_nones = False, else we return Nones too.
    while True:
      if self._next_returned_index == self._dataset_length:
        break
      with self._lock, timer:
        if (
            self._target_prefetch_buffer_size > 0
            and self._target_num_threads > 0
        ):
          if not self._buffer:
            # Fill the buffer on the first iteration.
            self._fill_buffer()
          self._measure_prefetch_depth()
          element = self._buffer.popleft()
          # Prefetch elements until the buffer is full again.
          self._fill_buffer()
          element = element.result()
        else:
          # In case prefetch buffer size was decreased, we still want to consume
          # the already prefetched elements.
          if self._buffer:
            element = self._buffer.popleft().result()
          else:
            element = self._stats.record_bytes_consumed(
                self._map_parent[self._next_returned_index]
            )
            self._next_buffered_index += 1
        self._next_returned_index += 1
      return_element = self._allow_nones or element is not None
      self._threshold_checker.check(return_element)
      if return_element:
        with self._stats.record_self_time(offset_ns=timer.value()):
          element = self._stats.record_bytes_produced(element)
          return self._stats.record_output_spec(element)
    raise StopIteration

  def get_state(self):
    return {"next_index": self._next_returned_index}

  def set_state(self, state):
    with self._lock:
      self._next_returned_index = state["next_index"]
      self._next_buffered_index = self._next_returned_index
      if (
          self._next_returned_index < 0
          or self._next_returned_index > self._dataset_length
      ):
        raise IndexError(
            f"Checkpoint `next_index` {self._next_returned_index} is out of"
            f" range for dataset of length {self._dataset_length}."
        )

      # Cancel all pending futures in the buffer.
      while self._buffer:
        future = self._buffer.popleft()
        future.cancel()

  def _get_next_index(self) -> int:
    return self._next_returned_index

  def _set_next_index(self, index: int) -> None:
    self.set_state({"next_index": index})

  def __str__(self) -> str:
    return (
        f"PrefetchDatasetIterator(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )

  def set_executor_wrapper(
      self, wrapper: typing.Callable[[futures.Executor], futures.Executor]
  ):
    self._executor_wrapper = wrapper

  def _set_prefetch_buffer_size(self, buffer_size: int):
    self._target_prefetch_buffer_size = buffer_size
    # The executor is created in the constructor only if the prefetch buffer
    # size is greater than 0. If the user changes the prefetch buffer size, we
    # need to create or destroy the executor accordingly.
    if (
        self._target_prefetch_buffer_size > 0
        and self._target_num_threads > 0
        and not hasattr(self, "_executor")
    ):
      self._executor = futures.ThreadPoolExecutor(
          self._target_num_threads, thread_name_prefix="grain-prefetch"
      )
      if self._executor_wrapper:
        self._executor = self._executor_wrapper(self._executor)
    elif self._target_prefetch_buffer_size == 0 and hasattr(self, "_executor"):
      self._executor.shutdown()
      delattr(self, "_executor")

  def _set_num_threads(self, num_threads: int) -> None:
    self._target_num_threads = num_threads
    old_executor = None
    # Accounts for the case where the executor does not exit. This can
    # happen if the prefetch buffer size is set to 0.
    if hasattr(self, "_executor"):
      old_executor = self._executor
    if self._target_num_threads > 0 and self._target_prefetch_buffer_size > 0:
      self._executor = futures.ThreadPoolExecutor(
          self._target_num_threads, thread_name_prefix="grain-prefetch"
      )
      if self._executor_wrapper:
        self._executor = self._executor_wrapper(self._executor)
    elif hasattr(self, "_executor"):
      delattr(self, "_executor")
    if old_executor is not None:
      # Allows the old executor to finish running the tasks it was already
      # assigned asynchronously.
      old_executor.shutdown(wait=False)

  def _fill_buffer(self):
    while (
        len(self._buffer) < self._target_prefetch_buffer_size
        and self._next_buffered_index < self._dataset_length
    ):
      # Note that we trigger creation of `_stats` in this (single) thread, it is
      # important because the stats initialization is not thread-safe.
      self._buffer.append(
          self._executor.submit(
              functools.partial(_getitem, self._stats, self._map_parent),
              self._next_buffered_index,
          )
      )
      self._next_buffered_index += 1

  def start_prefetch(self):
    if self._target_prefetch_buffer_size > 0 and self._target_num_threads > 0:
      self._fill_buffer()

  def close(self) -> None:
    """Shuts down the thread pool executor and cancels all pending futures."""
    if self._closed:
      return
    self._closed = True
    # Shutdown the thread pool executor if it exists.
    if hasattr(self, "_executor"):
      self._executor.shutdown(wait=False)
      # Cancel all pending futures in the buffer.
      while self._buffer:
        future = self._buffer.popleft()
        future.cancel()


def get_dataset_options(ds: dataset.IterDataset) -> base.DatasetOptions:
  result = base.DatasetOptions()
  to_visit = [ds]
  while to_visit:
    parent = to_visit.pop()
    if isinstance(parent, dataset.WithOptionsIterDataset):
      result = result.merge(parent.options)
    to_visit.extend(parent.parents)  # pyrefly: ignore[bad-argument-type]
  return result


class ThreadPrefetchIterDataset(dataset.IterDataset[T]):
  """Iterable dataset that uses a synchronized queue for prefetching.

  This is a thread-based alternative to `MultiprocessPrefetchIterDataset`.

  Attributes:
    parent: The parent dataset to prefetch from.
    prefetch_buffer_size: The size of the prefetch buffer. Must be greater than
      or equal to 0. If 0, prefetching is disabled and this is a noop.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      *,
      prefetch_buffer_size: int | grain_options.AutotuneParameter,
  ):
    super().__init__(parent)
    target_prefetch_buffer_size = prefetch_buffer_size
    if target_prefetch_buffer_size < 0:
      raise ValueError(
          "`prefetch_buffer_size` must be greater than or equal to 0, got "
          f"{target_prefetch_buffer_size}."
      )
    self._prefetch_buffer_size = prefetch_buffer_size

  def __str__(self) -> str:
    return (
        "ThreadPrefetchIterDataset("
        f"prefetch_buffer_size={self._prefetch_buffer_size})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return ThreadPrefetchDatasetIterator(
        parent_iter, self._prefetch_buffer_size
    )

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


# Type for the iterator state.
StateT = dict[str, Any]
# Type for the buffer elements.
BufferElementT = tuple[T, StateT, Exception | None]


class _PrefetchStopped(Exception):
  """Internal signal that thread prefetch was stopped and the buffer was closed.

  Placed on the buffer to wake a consumer blocked in ``Queue.get`` when
  cancellation is requested. Not meant to be raised to user code after an
  explicit ``close()``.
  """


def _buffer_put(
    buffer: queue.Queue[BufferElementT],
    item: BufferElementT,
    should_stop: threading.Event,
) -> None:
  """Puts ``item`` on ``buffer``, never blocking after a stop request.

  A stop sentinel may already occupy the only free slot. Blocking put after
  cancellation would deadlock with close(), which joins this thread.
  """
  if should_stop.is_set():
    try:
      buffer.put_nowait(item)
    except queue.Full:
      pass
    return
  while True:
    if should_stop.is_set():
      try:
        buffer.put_nowait(item)
      except queue.Full:
        pass
      return
    try:
      buffer.put(item, timeout=0.05)
      return
    except queue.Full:
      continue


def _put_iterator_elements_in_buffer(
    iterator: dataset.DatasetIterator[T],
    buffer: queue.Queue[BufferElementT],
    should_stop: threading.Event,
    stats: dataset_stats.Stats,
):
  """Fetches elements from the iterator and puts them in the buffer."""
  try:
    while not should_stop.is_set():
      element = stats.record_bytes_consumed(iterator.__next__())
      state = copy.deepcopy(iterator.get_state())
      # Re-check stop before a potentially blocking put so cancellation can
      # discard the element rather than wait for buffer space.
      if should_stop.is_set():
        return
      _buffer_put(buffer, (element, state, None), should_stop)
  except Exception as e:  # pylint: disable=broad-except
    _buffer_put(
        buffer, (None, None, e), should_stop
    )  # pyrefly: ignore[bad-argument-type]


def _request_stop_iterator_tree(iterator: dataset.DatasetIterator) -> None:
  """Non-blocking cancel propagation through a parent iterator chain.

  ThreadPrefetch nodes get ``request_stop()``. Other nodes are traversed without
  setting ``_closed`` on intermediate transforms, so a later ``close()`` can
  still walk the chain and join nested prefetch threads. Leaf iterators (no
  parents) are marked closed so a cooperative ``__next__`` can unblock.
  """
  if isinstance(iterator, ThreadPrefetchDatasetIterator):
    iterator.request_stop()
    return
  parents = iterator._parents  # pylint: disable=protected-access
  if not parents:
    iterator._closed = True  # pylint: disable=protected-access
    return
  for parent in parents:
    _request_stop_iterator_tree(parent)


class CheckpointableIterator(Iterator[T], Protocol[T]):
  """Iterator that can be checkpointed."""

  def get_state(self) -> StateT:
    """Returns the current state of the iterator."""

  def set_state(self, state: StateT):
    """Sets the current state of the iterator."""


class ThreadPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a synchronized queue."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: CheckpointableIterator[T],
      prefetch_buffer_size: int | grain_options.AutotuneParameter,
  ):
    if isinstance(parent, dataset.DatasetIterator):
      super().__init__(parent)
    else:
      super().__init__()
    self._maybe_nonnative_parent = parent

    target_prefetch_buffer_size = prefetch_buffer_size
    autotune_buffer_size = None

    assert target_prefetch_buffer_size >= 0, target_prefetch_buffer_size
    self._target_prefetch_buffer_size = target_prefetch_buffer_size
    self.autotune_buffer_size = autotune_buffer_size
    self._state: StateT | None = None
    self._next_index: int | None = 0

    self._prefetch_thread: threading.Thread | None = None
    self._prefetch_should_stop: threading.Event = threading.Event()
    if self.autotune_buffer_size is not None:
      self._buffer: (
          variable_size_queue.VariableSizeQueue | queue.Queue[BufferElementT]
      ) = variable_size_queue.VariableSizeQueue(
          max_size=self._target_prefetch_buffer_size
      )
    else:
      self._buffer: (
          variable_size_queue.VariableSizeQueue | queue.Queue[BufferElementT]
      ) = queue.Queue(maxsize=self._target_prefetch_buffer_size)

  # pytype: disable=attribute-error
  # pylint: disable=protected-access

  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ):
    # This method is needed to set `is_prefetch` to `True` in the stats config.
    parent_stats = [
        p._initialize_stats(execution_tracking_mode) for p in self._parents
    ]
    self._stats = _initialize_prefetch_stats(
        self, execution_tracking_mode, parent_stats
    )
    return self._stats

  @functools.cached_property
  def _stats(self):
    return self._initialize_stats(
        self._ctx.dataset_options.execution_tracking_mode
    )

  # pytype: enable=attribute-error
  # pylint: enable=protected-access

  def start_prefetch(self):
    """Starts prefetching elements in background.

    Raises:
      ValueError: If the iterator has been closed.
    """
    if self._closed:
      raise ValueError("Attempting to use a closed iterator.")
    if self._state is None:
      self._state = self._maybe_nonnative_parent.get_state()
    if self._prefetch_thread is not None:
      return

    self._prefetch_should_stop.clear()

    self._prefetch_thread = threading.Thread(
        target=functools.partial(
            _put_iterator_elements_in_buffer,
            iterator=self._maybe_nonnative_parent,
            buffer=self._buffer,
            should_stop=self._prefetch_should_stop,
            stats=self._stats,
        ),
        daemon=True,
        name=f"grain-thread-prefetch-{str(self)}",
    )
    self._prefetch_thread.start()

  @dataset_stats.record_next_duration_if_output
  @dataset_stats.trace_input_pipeline_next(
      stage_category=dataset_stats.IPL_CAT_PREFETCH
  )
  def __next__(self):
    # Check closed before any buffer read. A stop sentinel left in the queue
    # (especially with unbounded maxsize=0) must not surface as StopIteration
    # after close(); the documented contract is ValueError.
    if self._closed:
      raise ValueError("Attempting to use a closed iterator.")

    if self._state is None:
      self._state = self._maybe_nonnative_parent.get_state()

    timer = dataset_stats.Timer()
    with timer:
      if self._target_prefetch_buffer_size > 0:
        self.start_prefetch()
        element, state, err = self._buffer_get()
      else:
        try:
          # In case of 0 prefetch buffer size, we still try to get from the
          # buffer as it could have been populated when the prefetch buffer size
          # was greater than 0.
          element, state, err = self._buffer.get_nowait()
        except queue.Empty:
          element = self._maybe_nonnative_parent.__next__()
          state = copy.deepcopy(self._maybe_nonnative_parent.get_state())
          err = None

    if err is not None:
      # A stop sentinel means cancellation was already requested (possibly by a
      # parent close). Do not join here: the closing thread owns the join, and
      # joining from a nested producer can deadlock with close().
      if isinstance(err, _PrefetchStopped):
        raise StopIteration from err
      self._stop_prefetch()
      raise err
    self._state = state
    if self._next_index is not None:
      self._next_index += 1
    with self._stats.record_self_time(offset_ns=timer.value()):
      element = self._stats.record_bytes_produced(element)
      return self._stats.record_output_spec(element)

  def request_stop(self) -> None:
    """Requests cancellation without waiting for producer threads.

    Marks this iterator closed, signals the local prefetch thread, wakes any
    reader blocked on the buffer, and propagates the same non-blocking request
    down the parent chain. Does not join. Explicit ``close()`` joins after
    requesting stop. Safe to call from ``__del__``.
    """
    already_closed = self._closed
    self._closed = True
    self._request_stop_prefetch()
    if already_closed:
      return
    parent = self._maybe_nonnative_parent
    if isinstance(parent, dataset.DatasetIterator):
      _request_stop_iterator_tree(parent)

  def close(self):
    """Stops the iterator. No further calls to the iterator are expected.

    Cancellation is two-phase so a producer blocked in ``parent.__next__`` can
    be woken by closing the parent before this iterator joins its thread:

    1. ``request_stop()``: non-blocking cancel on this node and parents.
    2. ``parent.close()``: blocking cleanup of the parent chain (joins nested
       prefetch threads).
    3. Join the local prefetch thread (skipped while the interpreter finalizes).
    """
    self.request_stop()
    if isinstance(self._maybe_nonnative_parent, dataset.DatasetIterator):
      self._maybe_nonnative_parent.close()
    # Parent close may have unblocked our producer mid-put; clear again so the
    # join cannot stall on a full buffer. Re-issue the stop sentinel afterwards
    # and do not clear after join: a nested reader may still be blocked in
    # buffer.get(), and a post-join clear would steal the sentinel from it.
    self._clear_buffer()
    self._put_stop_sentinel()
    self._join_prefetch_thread(clear_buffer=False)

  def __del__(self):
    # Best-effort only: propagate non-blocking cancel, never join. Explicit
    # close() is the deterministic path. Joining from a finalizer can hang if
    # the producer is blocked in parent.__next__ or during interpreter
    # shutdown. request_stop() still marks parents closed so a producer parked
    # in parent.__next__ can observe cancel and drop its reference.
    try:
      self.request_stop()
    except Exception:  # pylint: disable=broad-except
      pass

  def _clear_buffer(self):
    while True:
      try:
        self._buffer.get_nowait()
      except queue.Empty:
        return

  def _buffer_get(self) -> BufferElementT:
    """Gets the next buffer item, honouring cancellation.

    The healthy path uses a blocking ``get`` (same as before this change) so a
    busy producer is not paced by a poll interval. After stop is requested,
    uses a short timeout so a nested reader can observe that this iterator was
    closed or that its prefetch thread finished even if a stop sentinel was
    drained by another closer. A stop sentinel still wakes a blocking get.
    """
    while True:
      if self._closed or self._prefetch_should_stop.is_set():
        try:
          return self._buffer.get(timeout=0.05)
        except queue.Empty:
          if self._closed:
            raise StopIteration
          thread = self._prefetch_thread
          if thread is None or not thread.is_alive():
            raise StopIteration
          continue
      # Running: park until an element or a stop sentinel arrives.
      return self._buffer.get()

  def _put_stop_sentinel(self):
    """Wakes a consumer blocked in ``buffer.get`` after a stop request."""
    sentinel: BufferElementT = (None, None, _PrefetchStopped())
    try:
      self._buffer.put_nowait(sentinel)
    except queue.Full:
      try:
        self._buffer.get_nowait()
      except queue.Empty:
        pass
      try:
        self._buffer.put_nowait(sentinel)
      except queue.Full:
        pass

  def _request_stop_prefetch(self, clear_buffer: bool = True):
    """Non-blocking cancellation request for the prefetch thread."""
    if self._prefetch_thread is None:
      return

    self._prefetch_should_stop.set()
    if clear_buffer:
      # Remove entries from the buffer to unblock the producer, so that it
      # checks should_stop and exits.
      self._clear_buffer()
    else:
      assert isinstance(self._buffer, variable_size_queue.VariableSizeQueue)
      # Increase the buffer size by 1 to unblock the producer.
      self._buffer.set_max_size(self._target_prefetch_buffer_size + 1)  # pytype: disable=attribute-error
    # Wake any reader blocked in get() (for example a parent producer in a
    # nested ThreadPrefetch pipeline).
    self._put_stop_sentinel()

  def _join_prefetch_thread(self, clear_buffer: bool = True):
    """Waits for the prefetch thread to exit after a stop request."""
    if self._prefetch_thread is None:
      return

    if not sys.is_finalizing():
      # Joining the worker thread is not necessary when the Python interpreter
      # is shutting down. Attempting to join can lead to hanging in Python
      # 3.13 as daemon threads can hang during interpreter shutdown. See
      # https://github.com/python/cpython/issues/123940#issuecomment-2976446309
      if self._prefetch_thread is not None:
        self._prefetch_thread.join()
    self._prefetch_thread = None

    if clear_buffer:
      # Clear the buffer again in case the prefetch loop added more elements
      # on exit.
      self._clear_buffer()

  def _stop_prefetch(self, clear_buffer: bool = True):
    """Stops the prefetching thread if it's currently running."""
    self._request_stop_prefetch(clear_buffer=clear_buffer)
    self._join_prefetch_thread(clear_buffer=clear_buffer)

  def get_state(self) -> StateT:
    if self._state is not None:
      return self._state
    else:
      # This point is only reached if `get_state` is called after
      # `set_next_index` and before the next `__next__` call. The prefetch
      # thread is not running at this point, so it is safe to call `get_state`
      # on the parent iterator.
      self._state = self._maybe_nonnative_parent.get_state()
      return self._state

  def set_state(self, state: StateT):
    self._stop_prefetch()
    self._maybe_nonnative_parent.set_state(state)
    self._state = self._maybe_nonnative_parent.get_state()
    self._next_index = None

  def _get_next_index(self) -> int:
    if self._next_index is not None:
      return self._next_index
    if not isinstance(self._maybe_nonnative_parent, dataset.DatasetIterator):
      raise ValueError(
          "`_get_next_index` only supported for native dataset iterators."
      )
    # This point is only reached if `set_state` and `get_next_index are called
    # on the same iterator. We need to get the index from the parent iterator
    # after setting the state to the point before all current buffer elements
    # were produced from the parent iterator.
    state = self.get_state()
    self._stop_prefetch()
    self._maybe_nonnative_parent.set_state(state)
    self._next_index = dataset.get_next_index(self._maybe_nonnative_parent)
    return self._next_index

  def _set_next_index(self, next_index: int):
    if not isinstance(self._maybe_nonnative_parent, dataset.DatasetIterator):
      raise ValueError(
          "`set_next_index` only supported for native dataset iterators."
      )
    self._stop_prefetch()
    dataset.set_next_index(self._maybe_nonnative_parent, next_index)
    self._next_index = next_index
    self._state = None

  def __str__(self) -> str:
    return (
        "ThreadPrefetchDatasetIterator("
        f"prefetch_buffer_size={self._target_prefetch_buffer_size})"
    )


class _MpContextIterDataset(dataset.IterDataset[T]):
  """Sets mp_context on iterator."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      mp_context: base.MultiprocessingContext,
  ):
    super().__init__(parent)
    self._mp_context = mp_context

  def __iter__(self) -> dataset.DatasetIterator[T]:
    it = self._parent.__iter__()
    it._ctx.mp_context = self._mp_context
    return it

  def __str__(self) -> str:
    return f"_MpContextIterDataset(mp_context={self._mp_context})"

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


def multithread_prefetch(
    ds: dataset.IterDataset[T],
    num_threads: int,
    buffer_size: int,
    sequential_slice: bool = False,
) -> dataset.IterDataset[T]:
  """Uses a pool of threads to prefetch elements ahead of time.

  This is a thread-based alternative to `multiprocess_prefetch`
  intended to be used with free-threaded Python.

  It works by sharding the input dataset into `num_threads` shards, and
  interleaving them. Each shard is read by a separate thread inside
  `InterleaveIterDataset`.

  Args:
    ds: The parent dataset to prefetch from.
    num_threads: The number of threads to use for prefetching. If 0, prefetching
      is disabled and this is a no-op.
    buffer_size: The size of the prefetch buffer for each thread.
    sequential_slice: Whether to use sequential slicing.

  Returns:
    An `IterDataset` that prefetches elements from `ds` using multiple threads.
  """
  if num_threads == 0:
    return ds

  dataset_options = get_dataset_options(ds)

  shards = []
  for i in range(num_threads):
    if num_threads == 1:
      worker_ds = ds
    else:
      worker_ds = copy.deepcopy(ds)
      dataset.set_slice(
          worker_ds, slice(i, None, num_threads), sequential_slice
      )
    shards.append(
        _MpContextIterDataset(
            worker_ds,
            base.MultiprocessingContext(
                process_index=i,
                process_count=num_threads,
            ),
        )
    )

  ds = interleave.InterleaveIterDataset(
      shards, cycle_length=num_threads, iter_buffer_size=buffer_size
  )
  # Apply options from parent dataset because interleave dataset does not
  # propagate options.
  ds = dataset.WithOptionsIterDataset(ds, dataset_options)
  return ds


def is_prefetch_iterator(it: dataset.DatasetIterator) -> bool:
  """Returns whether the iterator is a prefetch iterator."""
  # Loaded lazily due to a circular dependency (prefetch <-> process_prefetch).
  # pylint: disable=g-import-not-at-top
  from grain._src.python.dataset.transformations import process_prefetch
  # pylint: enable=g-import-not-at-top

  return isinstance(
      it,
      (
          PrefetchDatasetIterator,
          ThreadPrefetchDatasetIterator,
          interleave.InterleaveDatasetIterator,
          process_prefetch.ProcessPrefetchDatasetIterator,
      ),
  )
