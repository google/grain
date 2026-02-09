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
import threading
import typing
from typing import Any, Optional, Protocol, TypeVar

from concurrent import futures
from grain._src.core import monitoring as grain_monitoring
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import filter as filter_dataset
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import source

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


@typing.runtime_checkable
class SupportsInPlaceSlicing(Protocol):
  """Datasets that support mutation by setting the processed data slice."""

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    ...


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
      _set_slice_map_dataset(self._parent, sl, sequential_slice)

  def __str__(self) -> str:
    return (
        f"PrefetchIterDataset(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return PrefetchDatasetIterator(
        self._parent, self._read_options, self._allow_nones
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
    self._prefetch_buffer_size = (
        read_options.prefetch_buffer_size if read_options.num_threads > 0 else 0
    )
    self._num_threads = read_options.num_threads
    self._allow_nones = allow_nones
    if self._prefetch_buffer_size > 0:
      self._executor = futures.ThreadPoolExecutor(
          self._num_threads, thread_name_prefix="grain-prefetch"
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
        self._ctx.dataset_options.execution_tracking_mode
    )

  @functools.cached_property
  def _threshold_checker(self):
    # Sparse `MapDataset` transformations produce Nones which we filter out
    # here. The validator helps to detect if we discard too many elements.
    return filter_dataset.FilterThresholdChecker(
        transform_name=str(self),
        warn_threshold=self._ctx.dataset_options.filter_warn_threshold_ratio,
        raise_threshold=self._ctx.dataset_options.filter_raise_threshold_ratio,
    )

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
        if self._prefetch_buffer_size > 0:
          if not self._buffer:
            # Fill the buffer on the first iteration.
            self._fill_buffer()
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
      if self._prefetch_buffer_size > 0:
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

  def set_prefetch_buffer_size(self, buffer_size: int):
    self._prefetch_buffer_size = buffer_size
    # The executor is created in the constructor only if the prefetch buffer
    # size is greater than 0. If the user changes the prefetch buffer size, we
    # need to create or destroy the executor accordingly.
    if self._prefetch_buffer_size > 0 and not hasattr(self, "_executor"):
      if self._num_threads == 0:
        raise ValueError(
            "num_threads must be greater than 0 when prefetch buffer size is"
            " greater than 0."
        )
      self._executor = futures.ThreadPoolExecutor(
          self._num_threads, thread_name_prefix="grain-prefetch"
      )
    elif self._prefetch_buffer_size == 0 and hasattr(self, "_executor"):
      self._executor.shutdown()
      delattr(self, "_executor")

  def set_num_threads(self, num_threads: int) -> None:
    self._num_threads = num_threads
    old_executor = None
    # Accounts for the case where the executor does not exit. This can
    # happen if the prefetch buffer size is set to 0.
    if hasattr(self, "_executor"):
      old_executor = self._executor
    if self._num_threads > 0:
      self._executor = futures.ThreadPoolExecutor(
          self._num_threads, thread_name_prefix="grain-prefetch"
      )
    else:
      delattr(self, "_executor")
    if old_executor is not None:
      # Allows the old executor to finish running the tasks it was already
      # assigned asynchronously.
      old_executor.shutdown(wait=False)

  def _fill_buffer(self):
    while (
        len(self._buffer) < self._prefetch_buffer_size
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
    if self._prefetch_buffer_size > 0:
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


def _set_slice_iter_dataset(
    ds: dataset.IterDataset,
    sl: slice,
    sequential_slice: bool = False,
) -> None:
  """Sets data slice for the given dataset.IterDataset in place.

  WARNING: mutates the dataset object. Must only be used on dataset object copy.

  Applies recursively for parents.

  Args:
   ds: dataset.IterDataset to apply slice to.
   sl: slice to apply.
   sequential_slice: whether to apply sequential slicing.
  """
  if isinstance(ds, SupportsInPlaceSlicing):
    ds.set_slice(sl, sequential_slice)
    return
  if not ds.parents:
    raise ValueError(f"Cannot slice `IterDataset` source. {type(ds)}")
  for parent in ds.parents:
    if isinstance(parent, dataset.MapDataset):
      _set_slice_map_dataset(parent, sl, sequential_slice)
    else:
      _set_slice_iter_dataset(parent, sl, sequential_slice)


def _set_slice_map_dataset(
    ds: dataset.MapDataset,
    sl: slice,
    sequential_slice: bool = False,
) -> None:
  """Sets data slice for the given dataset.MapDataset in place.

  WARNING: mutates the dataset object. Must only be used on dataset object copy.

  Applies recursively for parents.

  Args:
   ds: dataset.MapDataset to apply slice to.
   sl: slice to apply.
   sequential_slice: whether to apply sequential slicing.
  """
  if isinstance(ds, SupportsInPlaceSlicing):
    ds.set_slice(sl, sequential_slice)
    return
  if not ds.parents:
    raise ValueError(f"Cannot slice `MapDataset` source. {type(ds)}")
  for parent in ds.parents:
    if isinstance(parent, dataset.MapDataset):
      _set_slice_map_dataset(parent, sl, sequential_slice)
    else:
      _set_slice_iter_dataset(parent, sl, sequential_slice)


def _get_dataset_options(ds: dataset.IterDataset) -> base.DatasetOptions:
  result = base.DatasetOptions()
  to_visit = [ds]
  while to_visit:
    parent = to_visit.pop()
    if isinstance(parent, dataset.WithOptionsIterDataset):
      result = result.merge(parent.options)
    to_visit.extend(parent.parents)
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
      prefetch_buffer_size: int,
  ):
    super().__init__(parent)
    if prefetch_buffer_size < 0:
      raise ValueError(
          "`prefetch_buffer_size` must be greater than or equal to 0, got "
          f"{prefetch_buffer_size}."
      )
    self._prefetch_buffer_size = prefetch_buffer_size

  def __str__(self) -> str:
    return (
        "ThreadPrefetchIterDataset("
        f"prefetch_buffer_size={self._prefetch_buffer_size})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    if self._prefetch_buffer_size == 0:
      return parent_iter
    return ThreadPrefetchDatasetIterator(
        parent_iter, self._prefetch_buffer_size
    )

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


# Type for the iterator state.
StateT = dict[str, Any]


def _put_iterator_elements_in_buffer(
    iterator: dataset.DatasetIterator[T],
    buffer: queue.Queue[tuple[T, StateT, Exception | None]],
    should_stop: threading.Event,
    stats: dataset_stats.Stats,
):
  """Fetches elements from the iterator and puts them in the buffer."""
  try:
    while not should_stop.is_set():
      element = stats.record_bytes_consumed(iterator.__next__())
      state = iterator.get_state()
      buffer.put((element, state, None))
  except Exception as e:  # pylint: disable=broad-except
    buffer.put((None, None, e))


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
      prefetch_buffer_size: int,
  ):
    if isinstance(parent, dataset.DatasetIterator):
      super().__init__(parent)
    else:
      super().__init__()
    self._maybe_nonnative_parent = parent

    assert prefetch_buffer_size > 0, prefetch_buffer_size
    self._prefetch_buffer_size = prefetch_buffer_size
    self._step_zero_state: StateT = parent.get_state()
    self._state: StateT | None = self._step_zero_state
    self._next_index: int | None = 0

    self._prefetch_thread: threading.Thread | None = None
    self._prefetch_should_stop: threading.Event = threading.Event()
    self._buffer: queue.Queue[tuple[T, StateT, Exception | None]] = queue.Queue(
        maxsize=self._prefetch_buffer_size
    )

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
    timer = dataset_stats.Timer()
    with timer:
      self.start_prefetch()
      element, state, err = self._buffer.get()

    if err is not None:
      self._stop_prefetch()
      raise err
    self._state = state
    if self._next_index is not None:
      self._next_index += 1
    with self._stats.record_self_time(offset_ns=timer.value()):
      element = self._stats.record_bytes_produced(element)
      return self._stats.record_output_spec(element)

  def close(self):
    """Stops the iterator. No further calls to the iterator are expected."""
    self._closed = True
    self._stop_prefetch()
    if isinstance(self._maybe_nonnative_parent, dataset.DatasetIterator):
      self._maybe_nonnative_parent.close()

  def _clear_buffer(self):
    while True:
      try:
        self._buffer.get_nowait()
      except queue.Empty:
        return

  def _stop_prefetch(self):
    """Stops the prefetching thread if it's currently running."""
    if self._prefetch_thread is None:
      return

    self._prefetch_should_stop.set()
    # Remove entries from the buffer to unblock the producer, so that it checks
    # producer_running.is_set() and exits.
    self._clear_buffer()
    self._prefetch_thread.join()
    self._prefetch_thread = None
    # Clear the buffer again in case the prefetch loop added more elements on
    # exit.
    self._clear_buffer()

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
        f"prefetch_buffer_size={self._prefetch_buffer_size})"
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

  dataset_options = _get_dataset_options(ds)

  shards = []
  for i in range(num_threads):
    if num_threads == 1:
      worker_ds = ds
    else:
      worker_ds = copy.deepcopy(ds)
      _set_slice_iter_dataset(
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
  return isinstance(
      it,
      (
          PrefetchDatasetIterator,
          ThreadPrefetchDatasetIterator,
          interleave.InterleaveDatasetIterator,
      ),
  )
