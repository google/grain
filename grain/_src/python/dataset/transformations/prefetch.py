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
import sys
import threading
import time
import typing
from typing import Any, Generic, Mapping, Optional, Protocol, TypeVar

import cloudpickle
from concurrent import futures
from grain._src.core import tree
import multiprocessing as mp
from grain._src.python import grain_pool
from grain._src.python import options as grain_options
from grain._src.python import shared_memory_array
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import filter as filter_dataset
import numpy as np

T = TypeVar("T")


@typing.runtime_checkable
class SupportsInPlaceSlicing(Protocol):
  """Datasets that support mutation by setting the processed data slice."""

  def set_slice(self, sl: slice) -> None:
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

  def set_slice(self, sl: slice) -> None:
    """Replaces `MapDataset` parents with their sliced versions."""
    assert isinstance(self._parent, dataset.MapDataset), self._parent
    self._parents = (self._parent.slice(sl),)
    self._parent._stats._is_output = False  # pylint: disable=protected-access

  def __str__(self) -> str:
    return (
        f"PrefetchIterDataset(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return PrefetchDatasetIterator(
        self._parent, self._read_options, self._allow_nones
    )


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
    self._next_index = 0
    self._buffer = None
    self._lock = threading.Lock()
    self._prefetch_buffer_size = read_options.prefetch_buffer_size
    self._allow_nones = allow_nones
    if self._prefetch_buffer_size > 0:
      self._executor = futures.ThreadPoolExecutor(read_options.num_threads)

  @functools.cached_property
  def _stats(self):
    execution_tracking_mode = self._options_with_default.execution_tracking_mode
    parent_stats = self._map_parent._initialize_stats(  # pylint: disable=protected-access
        execution_tracking_mode
    )
    # Connect to `MapDataset` parent stats.
    return dataset_stats.make_stats(
        dataset_stats.StatsConfig(
            name=str(self),
            transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
        ),
        (parent_stats,),
        execution_tracking_mode,
    )

  @functools.cached_property
  def _threshold_checker(self):
    # Sparse `MapDataset` transformations produce Nones which we filter out
    # here. The validator helps to detect if we discard too many elements.
    return filter_dataset.FilterThresholdChecker(
        transform_name=str(self),
        warn_threshold=self._options_with_default.filter_warn_threshold_ratio,
        raise_threshold=self._options_with_default.filter_raise_threshold_ratio,
    )

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
            # Stats initialization is not thread-safe, so we trigger map parent
            # stats initialization before multithreaded prefetching.
            _ = self._stats
            self._buffer = collections.deque(
                self._executor.submit(self._map_parent.__getitem__, i)
                for i in indices
            )
          element = self._buffer.popleft()
          if (
              self._next_index + self._prefetch_buffer_size
              < self._dataset_length
          ):
            self._buffer.append(
                self._executor.submit(
                    self._map_parent.__getitem__,
                    self._next_index + self._prefetch_buffer_size,
                )
            )
          element = element.result()
        else:
          element = self._map_parent[self._next_index]
        self._next_index += 1
      return_element = self._allow_nones or element is not None
      self._threshold_checker.check(return_element)
      if return_element:
        with self._stats.record_self_time(offset_ns=timer.value()):
          return element
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

  def __str__(self) -> str:
    return (
        f"PrefetchDatasetIterator(read_options={self._read_options},"
        f" allow_nones={self._allow_nones})"
    )


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
    if multiprocessing_options.num_workers < 0:
      raise ValueError(
          "`num_workers` must be greater than or equal to 0, got "
          f"{multiprocessing_options.num_workers}."
      )
    super().__init__(parent)
    self._multiprocessing_options = multiprocessing_options
    self._validate_parent_dataset()

  def __str__(self) -> str:
    return (
        "MultiprocessPrefetchIterDataset("
        f"multiprocessing_options={self._multiprocessing_options})"
    )

  def _validate_parent_dataset(self) -> None:
    """Checks the number of levels of parallelization."""
    to_check: list[dataset.MapDataset | dataset.IterDataset] = [self._parent]
    while to_check:
      ds = to_check.pop(0)
      if isinstance(ds, MultiprocessPrefetchIterDataset):
        raise ValueError(
            "Having multiple `MultiprocessPrefetchIterDataset`s is not "
            "allowed. Consider only keeping the last one."
        )
      to_check.extend(ds.parents)

  def __iter__(self) -> dataset.DatasetIterator[T]:
    if self._multiprocessing_options.num_workers == 0:
      return self._parent.__iter__()
    return MultiprocessPrefetchDatasetIterator(
        self._parent, self._multiprocessing_options
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


def _set_slice(ds: dataset.IterDataset, sl: slice) -> None:
  """Sets data slice for the given dataset in place.

  WARNING: mutates the dataset object. Must only be used on dataset object copy.

  Applies recursively for `IterDataset` parents.

  Args:
   ds: dataset to apply slice to.
   sl: slice to apply.
  """
  if isinstance(ds, SupportsInPlaceSlicing):
    ds.set_slice(sl)
    return
  if not ds.parents:
    raise ValueError("Cannot slice `IterDataset` source.")
  for parent in ds.parents:
    if isinstance(parent, dataset.MapDataset):
      raise NotImplementedError(
          "Slicing required by multiprocess prefetch is not implemented for"
          f" {ds}."
      )
    else:
      assert isinstance(parent, dataset.IterDataset), parent
      _set_slice(parent, sl)


def _check_picklable(
    ds: dataset.IterDataset | dataset.MapDataset,
):
  """Detects the first unpickle-able dataset in post-order.

  Args:
    ds: IterDataset or MapDataset to check whether it is picklable.

  NOTE: This function's time complexity is O(n^2) where n is the number of
  Grain dataset operations because `cloudpickle.dumps(ds)` will trigger
  pickling into all the datasets. If this naive O(n^2) algorithm takes too
  much time, we could consider doing copying `ds`, delete its parents and then
  do `cloudpickle.dumps(new_ds)` to reduce the time complexity to O(n).
  """

  # Traverses the graph in post-order to find the first unpickle-able subtree
  for parent in ds.parents:
    _check_picklable(parent)

  try:
    cloudpickle.dumps(ds)
  except Exception as e:  # pylint: disable=broad-exception-caught
    if sys.version_info >= (3, 11):
      e.add_note(
          f"Dataset: {ds} cannot be pickled!"
      )
    raise e


class GetElementProducerFn(grain_pool.GetElementProducerFn, Generic[T]):
  """Implements `GetElementProducerFn` for `grain_pool.MultiProcessIterator`.

  This class implements `GetElementProducerFn` with `serialize` being overriden
  to generate better error messages if user-provided dataset is not pickle-able.
  """

  def __init__(
      self, state: dict[str, dict[str, Any] | int], ds: dataset.IterDataset[T]
  ):
    self._state = state
    self._ds = ds

  def __call__(
      self, *, worker_index: int, worker_count: int
  ) -> Iterator[tuple[T, Optional[dict[str, Any]]]]:
    # Recover from the last recorded state for the given worker.
    worker_state = self._state[_WORKERS_STATE][str(worker_index)]
    if worker_count > 1:
      _set_slice(self._ds, slice(worker_index, None, worker_count))
    it = iter(self._ds)
    it.set_state(worker_state)  # pytype: disable=attribute-error
    # Skip the required number of iterations after the last recorded state.
    for _ in range(self._state[_ITERATIONS_TO_SKIP][str(worker_index)]):
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

  def serialize(self) -> bytes:
    """Overrides the default implementation to generate better error messages."""

    try:
      return cloudpickle.dumps(self)
    except Exception as e:  # pylint: disable=broad-except
      # Calls `_check_picklable` to generate useful pickle errors
      #
      # Note: No need to check `self._state` because it should not generate
      # unpicklable errors and it is controlled by us, not from user's code
      # in most cases. Except for the case when users try to implement their own
      # `MapDataset` and `IterDataset` with custom pickle-ing logic that
      # contains unpickle-able objects.
      _check_picklable(self._ds)

      # If somehow we cannot find the dataset that is causing the pickle
      # issues, just raise the original error
      raise e


class MultiprocessPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a multiprocessing pool."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      multiprocessing_options: grain_options.MultiprocessingOptions,
  ):
    super().__init__()
    self._iter_parent = parent
    self._multiprocessing_options = multiprocessing_options
    # The underlying iterator producing elements and workers state.
    self._iterator = None
    # Raw reference to the underlying iterator that can be used to determine the
    # last worker index.
    self._raw_iterator = None
    # Create initial state. We record state of each worker periodically together
    # with the number of iterations without the recorded state and index of the
    # last worker.
    workers_state: dict[str, Any] = {}
    iterations_to_skip: dict[str, int] = {}
    for i in range(multiprocessing_options.num_workers):
      workers_state[str(i)] = iter(
          self._iter_parent
      ).get_state()  # pytype: disable=attribute-error
      iterations_to_skip[str(i)] = 0

    self._state: dict[str, dict[str, Any] | int] = {
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

      # pytype: disable=annotation-type-mismatch
      iterations_to_skip: dict[str, Any] = self._state[_ITERATIONS_TO_SKIP]
      worker_state: dict[str, Any] = self._state[_WORKERS_STATE]
      # pytype: enable=annotation-type-mismatch

      self._state[_LAST_WORKER_INDEX] = worker_index
      worker_index_str = str(worker_index)
      if state is None:
        iterations_to_skip[worker_index_str] += 1
      else:
        iterations_to_skip[worker_index_str] = 0
        worker_state[worker_index_str] = state
    return _open_struct_from_shm(result)

  def start_prefetch(self) -> None:
    """Prefetches elements from the iterator.

    This will run background processes for prefetching. To make sure to clean up
    the resources, it should be followed by at least one `next` call.
    """
    self._ensure_iterator_initialized()

  def set_state(self, state: dict[str, dict[str, Any] | int]) -> None:
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

    get_element_producer_fn = GetElementProducerFn(
        self._state, self._iter_parent
    )

    return grain_pool.MultiProcessIterator(
        get_element_producer_fn,
        self._multiprocessing_options,
        (self._state[_LAST_WORKER_INDEX] + 1)
        % self._multiprocessing_options.num_workers,
    )

  def __str__(self) -> str:
    return (
        "MultiprocessPrefetchDatasetIterator("
        f"multiprocessing_options={self._multiprocessing_options})"
    )


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
      # Avoid raising a NotImplemented error and make a noop instead.
      parent_iter.start_prefetch = lambda: None
      return parent_iter
    return _ThreadPrefetchDatasetIterator(
        parent_iter, self._prefetch_buffer_size, str(self)
    )


# Type for the iterator state.
StateT = Mapping[str, Any]


# Representation of the initial state, pre-next.
_INITIAL_STATE_SENTINEL = object()


class _ThreadPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a synchronized queue."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.DatasetIterator[T],
      prefetch_buffer_size: int,
      parent_transform_name: str,
  ):
    super().__init__(parent)
    assert prefetch_buffer_size > 0, prefetch_buffer_size
    self._prefetch_buffer_size = prefetch_buffer_size
    self._parent_transform_name = parent_transform_name
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
          target=self._work_loop,
          daemon=True,
          name=f"Prefetch-{self._parent_transform_name}",
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
        self._parent.set_state(initial_state)
      else:
        # Put the initial state of the iterator with a sentinel value, which
        # will be discarded. This avoids having to call a potentially expensive
        # and unused get_state() on the main thread.
        output_buffer.put(
            (_INITIAL_STATE_SENTINEL, self._parent.get_state(), None)
        )
      # Check if the producer thread should be running every time an item is
      # retrieved from the queue.
      while running.is_set():
        while True:
          element, state = next(self._parent), self._parent.get_state()
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

  def __str__(self) -> str:
    return (
        "ThreadPrefetchDatasetIterator("
        f"prefetch_buffer_size={self._prefetch_buffer_size})"
    )
