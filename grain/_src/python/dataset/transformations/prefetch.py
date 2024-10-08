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
from collections.abc import Callable, Iterator
import contextlib
import copy
import functools
import queue
import threading
import time
from typing import Any, Mapping, Optional, TypeVar

from concurrent import futures
import multiprocessing as mp
from grain._src.core import tree
from grain._src.python import grain_pool
from grain._src.python import options as grain_options
from grain._src.python import shared_memory_array
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
import numpy as np

T = TypeVar("T")


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

  def __str__(self) -> str:
    return (
        f"PrefetchIterDataset(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return PrefetchDatasetIterator(
        self._parent, self._read_options, self._allow_nones, self._stats
    )


class PrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a thread pool."""

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      read_options: grain_options.ReadOptions,
      allow_nones: bool,
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._dataset_length = len(parent)
    self._next_index = 0
    self._buffer = None
    self._lock = threading.Lock()
    self._prefetch_buffer_size = read_options.prefetch_buffer_size
    self._allow_nones = allow_nones
    if self._prefetch_buffer_size > 0:
      self._executor = futures.ThreadPoolExecutor(read_options.num_threads)

  def __next__(self) -> T:
    # We loop here to skip all None elements (in case the underlying dataset
    # is sparse), if self._allow_nones = False, else we return Nones too.
    timer = dataset_stats.Timer()
    while True:
      if self._next_index == self._dataset_length:
        break
      with self._lock, timer:
        if self._prefetch_buffer_size > 0:
          if not self._buffer:
            indices = range(
                self._next_index,
                min(
                    self._next_index + self._prefetch_buffer_size,
                    self._dataset_length,
                ),
            )
            self._buffer = collections.deque(
                self._executor.submit(self._parent.__getitem__, i)
                for i in indices
            )
          element = self._buffer.popleft()
          if (
              self._next_index + self._prefetch_buffer_size
              < self._dataset_length
          ):
            self._buffer.append(
                self._executor.submit(
                    self._parent.__getitem__,
                    self._next_index + self._prefetch_buffer_size,
                )
            )
          element = element.result()
        else:
          element = self._parent[self._next_index]
        self._next_index += 1
      if self._allow_nones or element is not None:
        with self._stats.record_self_time(offset_ns=timer.value()):
          return self._stats.record_output_spec(element)
    with self._stats.record_self_time(offset_ns=timer.value()):
      raise StopIteration

  def get_state(self):
    return {"next_index": self._next_index}

  def set_state(self, state):
    with self._lock:
      self._next_index = state["next_index"]
      if self._next_index < 0 or self._next_index > self._dataset_length:
        raise IndexError(
            f"Checkpoint `next_index` {self._next_index} is out of range for"
            f" dataset of length {self._dataset_length}."
        )
      if self._prefetch_buffer_size > 0:
        self._buffer = None


def _iterator_with_context(
    iterator: contextlib.AbstractContextManager[Iterator[T]],
) -> Iterator[T]:
  with iterator as it:
    yield from it


class MultiprocessPrefetchIterDataset(dataset.IterDataset[T]):
  """Uses a pool of processes to prefetch elements ahead of time.

  It usually makes sense to add this transformation in the end of the pipeline
  since it will execute the parent IterDataset in multiple processes.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      multiprocessing_options: grain_options.MultiprocessingOptions,
  ):
    if multiprocessing_options.num_workers < 1:
      raise ValueError(
          "`num_workers` must be greater than 0, got "
          f"{multiprocessing_options.num_workers}."
      )
    super().__init__(parent)
    self._validate_parent_dataset()
    self._multiprocessing_options = multiprocessing_options

  def __str__(self) -> str:
    return f"MultiprocessPrefetchIterDataset(multiprocessing_options={self._multiprocessing_options})"

  def _validate_parent_dataset(self):
    """Checks that there's a single level of parallelization."""
    to_check = [self._parent]
    while to_check:
      ds = to_check.pop(0)
      if isinstance(ds, MultiprocessPrefetchIterDataset):
        raise ValueError(
            "Having multiple `MultiprocessPrefetchIterDataset`s is not "
            "allowed. Consider only keeping the last one."
        )
      to_check.extend(ds.parents)

  def __iter__(self) -> MultiprocessPrefetchDatasetIterator[T]:
    return MultiprocessPrefetchDatasetIterator(
        self._parent, self._multiprocessing_options, self._stats
    )


# Keys in `MultiprocessPrefetchDatasetIterator` checkpoints.
_WORKERS_STATE = "workers_state"
_ITERATIONS_TO_SKIP = "iterations_to_skip"
_LAST_WORKER_INDEX = "last_worker_index"

# Minimal interval (in seconds) between consecutive state recordings in worker
# processes of `MultiprocessPrefetchDatasetIterator`. We record the state
# periodically to reduce the overhead of sending the state from workers.
# Note that this is also an approximate upper bound on how long it is going to
# take to recover from a checkpointed state. Larger values will decrease the
# overhead of sending the updated state but will also make recovery from a
# checkpoint longer on average.
_RECORD_STATE_INTERVAL_S = 3


def _copy_leaf_to_shm(leaf: Any) -> Any:
  """Copies `leaf` to shared memory if it's a numpy array."""
  if (
      not isinstance(leaf, np.ndarray)
      or leaf.dtype.hasobject
      or not leaf.flags.c_contiguous
  ):
    return leaf

  shared_memory_arr = shared_memory_array.SharedMemoryArray(
      leaf.shape, leaf.dtype
  )
  np.copyto(shared_memory_arr, leaf, casting="no")
  return shared_memory_arr.metadata


def _copy_struct_to_shm(struct: Any) -> Any:
  """Copies leaf ndarrays of the structure to shared memory."""
  return tree.map_structure(_copy_leaf_to_shm, struct)


def _open_leaf_from_shm(leaf: Any) -> Any:
  """Recovers `leaf` from shared memory if it's a numpy array metadata."""
  if isinstance(leaf, shared_memory_array.SharedMemoryArrayMetadata):
    leaf = shared_memory_array.SharedMemoryArray.from_metadata(leaf)
    leaf.unlink_on_del()
  return leaf


def _open_struct_from_shm(struct: Any) -> Any:
  """Recovers leaf ndarrays of the structure from shared memory."""
  return tree.map_structure(_open_leaf_from_shm, struct)


class MultiprocessPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a multiprocessing pool."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      multiprocessing_options: grain_options.MultiprocessingOptions,
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._multiprocessing_options = multiprocessing_options
    # The underlying iterator producing elements and workers state.
    self._iterator = None
    # Raw reference to the underlying iterator that can be used to determine the
    # last worker index.
    self._raw_iterator = None
    # Create initial state. We record state of each worker periodically together
    # with the number of iterations without the recorded state and index of the
    # last worker.
    workers_state = {}
    iterations_to_skip = {}
    for i in range(multiprocessing_options.num_workers):
      workers_state[str(i)] = iter(self._parent).get_state()  # pytype: disable=attribute-error
      iterations_to_skip[str(i)] = 0

    self._state = {
        _WORKERS_STATE: workers_state,
        _ITERATIONS_TO_SKIP: iterations_to_skip,
        _LAST_WORKER_INDEX: -1,
    }

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return self

  def __next__(self) -> T:
    self._ensure_iterator_initialized()
    result, state = next(self._iterator)
    with self._stats.record_self_time():
      worker_index = self._raw_iterator.get_last_worker_index()  # pytype: disable=attribute-error
      self._state[_LAST_WORKER_INDEX] = worker_index
      worker_index_str = str(worker_index)
      if state is None:
        self._state[_ITERATIONS_TO_SKIP][worker_index_str] += 1
      else:
        self._state[_ITERATIONS_TO_SKIP][worker_index_str] = 0
        self._state[_WORKERS_STATE][worker_index_str] = state
    return _open_struct_from_shm(self._stats.record_output_spec(result))

  def start_prefetch(self) -> None:
    """Prefetches elements from the iterator.

    This will run background processes for prefetching. To make sure to clean up
    the resources, it should be followed by at least one `next` call.
    """
    self._ensure_iterator_initialized()

  def set_state(self, state) -> None:
    self._state = state
    self._raw_iterator = None
    self._iterator = None

  def get_state(self) -> dict[str, Any]:
    return copy.deepcopy(self._state)

  def _ensure_iterator_initialized(self) -> None:
    if self._iterator is None:
      self._raw_iterator = self._create_iterator_context()
      self._raw_iterator.start_prefetch()
      self._iterator = _iterator_with_context(self._raw_iterator)

  def _create_iterator_context(self) -> grain_pool.MultiProcessIterator[T]:
    """Creates a `MultiProcessIterator`."""

    state = self._state
    parent = self._parent

    def get_element_producer_fn(
        worker_index: int, worker_count: int
    ) -> Iterator[tuple[T, Optional[dict[str, Any]]]]:
      # Recover from the last recorded state for the given worker.
      worker_state = state[_WORKERS_STATE][str(worker_index)]
      if worker_count > 1:
        parent._set_parent_maps_slice(slice(worker_index, None, worker_count))  # pylint: disable=protected-access
      it = iter(parent)
      it.set_state(worker_state)  # pytype: disable=attribute-error
      # Skip the required number of iterations after the last recorded state.
      for _ in range(state[_ITERATIONS_TO_SKIP][str(worker_index)]):
        _ = next(it)
      last_recorded_state_time = time.time()
      for element in it:
        now = time.time()
        element = _copy_struct_to_shm(element)
        if now - last_recorded_state_time >= _RECORD_STATE_INTERVAL_S:
          last_recorded_state_time = now
          yield (element, it.get_state())  # pytype: disable=attribute-error
        else:
          yield (element, None)

    return grain_pool.MultiProcessIterator(
        get_element_producer_fn,
        self._multiprocessing_options,
        (self._state[_LAST_WORKER_INDEX] + 1)
        % self._multiprocessing_options.num_workers,
    )


class ThreadPrefetchIterDataset(dataset.IterDataset[T]):
  """Iterable dataset that uses a synchronized queue for prefetching.

  This is a thread-based alternative to `MultiprocessPrefetchIterDataset`.

  Attributes:
    parent: The parent dataset to prefetch from.
    prefetch_buffer_size: The size of the prefetch buffer.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      *,
      prefetch_buffer_size: int,
  ):
    super().__init__(parent)
    self._prefetch_buffer_size = prefetch_buffer_size

  def __str__(self) -> str:
    return f"ThreadPrefetchIterDataset(prefetch_buffer_size={self._prefetch_buffer_size})"

  def __iter__(self) -> ThreadPrefetchDatasetIterator[T]:
    return ThreadPrefetchDatasetIterator(
        self._parent, self._prefetch_buffer_size, self._stats
    )


# Type for the iterator state.
StateT = Mapping[str, Any]


# Representation of the initial state, pre-next.
_INITIAL_STATE_SENTINEL = object()


class ThreadPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a synchronized queue."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      prefetch_buffer_size: int,
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent: dataset.IterDataset[T] = parent
    self._iterator: dataset.DatasetIterator[T] = parent.__iter__()
    self._prefetch_buffer_size = prefetch_buffer_size
    self._state: StateT | None = None

    self._work_queue = queue.Queue[Callable[[], Any]]()
    self._work_thread: threading.Thread | None = None
    # Whether this iterator is closed, meaning it should no longer be used.
    self._closed = False
    self._producer_running: threading.Event = None
    self._buffer: queue.Queue[tuple[T, StateT, Exception | None]] = None

  def _start_producer(self, initial_state: None):
    """Starts the producer.

    Args:
      initial_state: An optional initial state to set on the delegate.

    Raises:
      ValueError: If the iterator has been closed, or if the producer is already
        running.
    """
    if self._closed:
      raise ValueError("Attempting to use a closed iterator.")
    if self._producer_running is not None:
      raise ValueError("The producer is already running.")

    if self._work_thread is None:
      self._work_thread = threading.Thread(
          target=self._work_loop, daemon=True, name=f"Prefetch-{self._parent}"
      )
      self._work_thread.start()

    self._state = initial_state
    self._producer_running = threading.Event()
    self._producer_running.set()
    self._buffer = queue.Queue(maxsize=self._prefetch_buffer_size)
    self._work_queue.put(
        functools.partial(
            self._producer,
            initial_state=initial_state,
            output_buffer=self._buffer,
            running=self._producer_running,
        )
    )

  def _producer(
      self,
      initial_state,
      output_buffer: queue.Queue[tuple[T, StateT, Exception | None]],
      running: threading.Event,
  ) -> None:
    """Functor that fills the queue to its capacity.

    Should be run on a separate thread.

    Args:
      initial_state: state to initialize the itertor to.
      output_buffer: queue to fill.
      running: an sync event for whether the thread should run.
    """
    try:
      if initial_state is not None:
        self._iterator.set_state(initial_state)
      else:
        # Put the initial state of the iterator with a sentinel value, which
        # will be discarded. This avoids having to call a potentially expensive
        # and unused get_state() on the main thread.
        output_buffer.put(
            (_INITIAL_STATE_SENTINEL, self._iterator.get_state(), None)
        )
      # Check if the producer thread should be running every time an item is
      # retrieved from the queue.
      while running.is_set():
        while True:
          element, state = next(self._iterator), self._iterator.get_state()
          output_buffer.put((element, state, None))
          break
    except Exception as e:  # pylint: disable=broad-except
      output_buffer.put((None, None, e))

  def __next__(self):
    self.start_prefetch()
    assert self._buffer is not None
    element, state, err = self._buffer.get()

    if err is not None:
      raise err
    if self._state is None or element is _INITIAL_STATE_SENTINEL:
      # Both conditions should be simultaneously true and only once.
      assert element is _INITIAL_STATE_SENTINEL
      if self._state is not None:
        raise AssertionError(f"Expected {self._state=} to be None. {state=}.")
      self._state = state
      # Current call has retrieved a sentinel value and the initial state,
      # make another call to retrieve the actual first value from the delegate
      # iterator.
      return next(self)
    else:
      self._state = state
      return element

  def close(self):
    """Stops the iterator. No further calls to the iterator are expected."""
    self._closed = True
    self._stop_producer()
    # Make sure the work thread isn't blocked, so it can exit.
    self._work_queue.put(lambda: None)

  def start_prefetch(self):
    """Starts the producer if it's not already running.

    Raises:
      ValueError: If the iterator has been closed, or if there's already a
        running producer.
    """
    if self._closed:
      raise ValueError("Attempting to use a closed iterator.")
    if self._producer_running is None:
      self._start_producer(None)

  def _stop_producer(self):
    """Stops the producer if it's currently running."""
    producer_running = self._producer_running
    buffer = self._buffer
    if producer_running is None:
      # Nothing to stop.
      return

    producer_running.clear()
    # Remove entries from the buffer to unblock the producer, so that it checks
    # producer_running.is_set() and exits.
    assert buffer is not None  # PyType.
    while True:
      try:
        buffer.get_nowait()
      except queue.Empty:
        break
    self._producer_running = None
    self._buffer = None

  def get_state(self):
    self.start_prefetch()
    if self._state is None:
      # `__next__` has not been called, the first tuple in the buffer should be
      # made up of the `_INITIAL_STATE_SENTINEL` value and the initial state of
      # the delegate iterator.
      buffer = self._buffer
      assert buffer is not None  # PyType.
      val, state, err = buffer.get()
      if err is not None:
        raise err
      assert val is _INITIAL_STATE_SENTINEL
      assert state is not None
      self._state = state
    return self._state

  def set_state(self, state):
    self._stop_producer()
    self._state = state
    if self._prefetch_buffer_size > 0:
      self._buffer = None
    self._start_producer(state)

  def _work_loop(self):
    while not self._closed:
      self._work_queue.get()()
