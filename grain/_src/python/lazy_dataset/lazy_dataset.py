# Copyright 2023 Google LLC
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
"""LazyDataset base classes.

There are 3 main classes:
- `LazyMapDataset` define a dataset that supports efficient random access. It
  has 3 important properties:
  - `__len__()` returns the length of a single epoch over the dataset.
  - `__getitem__()` will return the element at any given (positive) index. The
    "true" length of a `LazyMapDataset` is infinite. Many implementations will
    simply loop but exceptions exists (e.g. `ShuffleLazyMapDataset` will loop
    with a different order).
  - The dataset is lazy and individual elements are only created when calling
    `__getitem__()`. Most `LazyMapDatasets`s are statements and will not hold
    elements.
- `LazyIterDataset` defines a dataset that does not support efficient random
  access. It can still be iterated over. A `LazyMapDataset` can be turned into
  a `LazyIterDataset` but going from `LazyIterDataset` to `LazyMapDataset` might
  be as expensive as materializing the whole dataset.
  A `LazyIterDataset` can have known, unknown or infinite length.
- `LazyDatasetIterator` defines a stateful iterator over `LazyIterDataset`. The
  state of the iterator can be saved and restored.

Using the interfaces defined in `collections.abc` you can think of
LazyMapDataset as (infinite) Sequence, LazyIterDataset as Iterable and
LazyDatasetIterator as Iterator.
"""
from __future__ import annotations

import abc
import collections
from collections.abc import Iterable, Iterator, Sequence
import contextlib
import copy
import functools
import time
from typing import Any, Callable, Optional, TypeVar, overload

from concurrent import futures
from grain._src.core import sharding
from grain._src.core import tree
from grain._src.core import usage_logging
from grain._src.python import grain_pool
from grain._src.python import options as grain_options
from grain._src.python import shared_memory_array
import numpy as np

T = TypeVar("T")
_MAX_PREFETCH_THREADS = 1000


class LazyMapDataset(Sequence[T], abc.ABC):
  """Abstract base class for all LazyMapDataset classes."""

  _functions: dict[str, Callable[[LazyMapDataset], Any]] = {}

  def __init__(self, parents: LazyMapDataset | Sequence[LazyMapDataset] = ()):
    if isinstance(parents, LazyMapDataset):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    usage_logging.log_event("LazyMapDataset", tag_3="PyGrain")

  @property
  def parents(self) -> Sequence[LazyMapDataset]:
    return self._parents

  @property
  def _parent(self) -> LazyMapDataset:
    assert len(self._parents) == 1
    return self._parents[0]

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the length of this dataset."""

  @overload
  def __getitem__(self, index: slice) -> LazyMapDataset:
    ...

  @overload
  def __getitem__(self, index: int) -> T | None:
    ...

  @abc.abstractmethod
  def __getitem__(self, index):
    """Returns the element for the index or None if missing."""

  @classmethod
  def register_function(
      cls, name: str, function: Callable[[LazyMapDataset], Any]
  ):
    if name in cls._functions:
      raise ValueError(
          f"Cannot register {function} as dataset function '{name}' since it's"
          f" already taken by {cls._functions[name]}."
      )
    cls._functions[name] = function

  def __getattr__(self, attribute_name: str):
    if attribute_name in LazyMapDataset._functions:
      return functools.partial(LazyMapDataset._functions[attribute_name], self)
    raise AttributeError(
        f"'{self.__class__.__name__}' object has no attribute"
        f" '{attribute_name}' :("
    )

  def __iter__(self) -> LazyDatasetIterator[T]:
    return self.to_iter_dataset().__iter__()

  def to_iter_dataset(
      self, read_options: grain_options.ReadOptions | None = None
  ) -> LazyIterDataset[T]:
    """Syntactic sugar to construct a LazyIterDataset."""
    return PrefetchLazyIterDataset(
        self, read_options=read_options or grain_options.ReadOptions()
    )


class LazyIterDataset(Iterable[T], abc.ABC):
  """Abstract base class for all LazyIterDataset classes."""

  _functions: dict[str, Callable[[LazyIterDataset], Any]] = {}

  def __init__(
      self,
      parents: (
          LazyMapDataset
          | LazyIterDataset
          | Sequence[LazyMapDataset | LazyIterDataset]
      ) = (),
  ):
    if isinstance(parents, (LazyMapDataset, LazyIterDataset)):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    usage_logging.log_event("LazyIterDataset", tag_3="PyGrain")

  @property
  def parents(self) -> Sequence[LazyMapDataset | LazyIterDataset]:
    return self._parents

  @property
  def _parent(self) -> LazyMapDataset | LazyIterDataset:
    assert len(self._parents) == 1, self._parents
    return self._parents[0]

  def set_parent_maps_slice(self, sl: slice) -> None:
    """Replaces LazyMapDataset-type parents with their sliced versions.

    Applies recursively for LazyIterDataset-type parents.

    Args:
     sl: slice to apply.
    """
    sliced_parents = []
    for parent in self._parents:
      if isinstance(parent, LazyMapDataset):
        sliced_parents.append(parent.slice(sl))
      else:
        parent.set_parent_maps_slice(sl)
        sliced_parents.append(parent)
    self._parents = tuple(sliced_parents)

  @abc.abstractmethod
  def __iter__(self) -> LazyDatasetIterator[T]:
    """Returns an iterator for this dataset."""

  @classmethod
  def register_function(
      cls, name: str, function: Callable[[LazyIterDataset], Any]
  ):
    if name in cls._functions:
      raise ValueError(
          f"Cannot register {function} as dataset function '{name}' since it's"
          f" already taken by {cls._functions[name]}."
      )
    cls._functions[name] = function

  def __getattr__(self, attribute_name: str):
    if attribute_name in LazyIterDataset._functions:
      return functools.partial(LazyIterDataset._functions[attribute_name], self)
    raise AttributeError(
        f"'{self.__class__.__name__}' object has no attribute"
        f" '{attribute_name}' :("
    )


def lazy_map_dataset_function(name: str):
  def _fn(cls):
    LazyMapDataset.register_function(name=name, function=cls)
    return cls

  return _fn


def lazy_iter_dataset_function(name: str):
  def _fn(cls):
    LazyIterDataset.register_function(name=name, function=cls)
    return cls

  return _fn


class LazyDatasetIterator(Iterator[T], abc.ABC):
  """Abstract base class for all LazyIterDataset iterator classes."""

  def __iter__(self) -> LazyDatasetIterator[T]:
    return self

  # __next__ abstract method since we inherit from Iterator[T].

  @abc.abstractmethod
  def get_state(self) -> dict[str, Any]:
    """Returns the current state of the iterator."""

  @abc.abstractmethod
  def set_state(self, state: dict[str, Any]):
    """Sets the current state of the iterator."""


@lazy_map_dataset_function("prefetch")
class PrefetchLazyIterDataset(LazyIterDataset[T]):
  """Iterable dataset that uses a thread pool for prefetching."""

  def __init__(
      self,
      parent: LazyMapDataset[T],
      *,
      read_options: grain_options.ReadOptions,
      allow_nones: bool = False,
  ):
    super().__init__(parent)
    self._read_options = read_options
    self._allow_nones = allow_nones

  def __iter__(self) -> LazyDatasetIterator[T]:
    return PrefetchLazyDatasetIterator(
        self._parent, self._read_options, self._allow_nones
    )


class PrefetchLazyDatasetIterator(LazyDatasetIterator[T]):
  """Iterator that performs prefetching using a thread pool."""

  def __init__(
      self,
      dataset: LazyMapDataset[T],
      read_options: grain_options.ReadOptions,
      allow_nones: bool,
  ):
    super().__init__()
    self._dataset = dataset
    self._dataset_length = len(dataset)
    self._next_index = 0
    self._buffer = None
    self._prefetch_buffer_size = read_options.prefetch_buffer_size
    self._allow_nones = allow_nones
    if self._prefetch_buffer_size > 0:
      self._executor = futures.ThreadPoolExecutor(read_options.num_threads)

  def __next__(self) -> T:
    # We loop here to skip all None elements (in case the underlying dataset
    # is sparse), if self._allow_sparsity = False, else we return Nones too.
    while True:
      if self._next_index == self._dataset_length:
        break
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
              self._executor.submit(self._dataset.__getitem__, i)
              for i in indices
          )
        element = self._buffer.popleft()
        if self._next_index + self._prefetch_buffer_size < self._dataset_length:
          self._buffer.append(
              self._executor.submit(
                  self._dataset.__getitem__,
                  self._next_index + self._prefetch_buffer_size,
              )
          )
        element = element.result()
      else:
        element = self._dataset[self._next_index]
      self._next_index += 1
      if self._allow_nones or element is not None:
        return element
    raise StopIteration

  def get_state(self):
    return {"next_index": self._next_index}

  def set_state(self, state):
    self._next_index = state["next_index"]
    if self._prefetch_buffer_size > 0:
      self._buffer = None


def _iterator_with_context(
    iterator: contextlib.AbstractContextManager[Iterator[T]],
) -> Iterator[T]:
  with iterator as it:
    yield from it


@lazy_iter_dataset_function("prefetch")
class MultiprocessPrefetchLazyIterDataset(LazyIterDataset[T]):
  """Uses a pool of processes to prefetch elements ahead of time.

  It usually makes sense to add this transformation in the end of the pipeline
  since it will execute the parent LazyIterDataset in multiple processes.
  """

  def __init__(
      self,
      parent: LazyIterDataset[T],
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

  def _validate_parent_dataset(self):
    """Checks that there's a single level of parallelization."""
    to_check = [self._parent]
    while to_check:
      dataset = to_check.pop(0)
      if isinstance(dataset, MultiprocessPrefetchLazyIterDataset):
        raise ValueError(
            "Having multiple `MultiprocessPrefetchLazyIterDataset`s is not "
            "allowed. Consider only keeping the last one."
        )
      to_check.extend(dataset.parents)

  def __iter__(self) -> LazyDatasetIterator[T]:
    return MultiprocessPrefetchLazyDatasetIterator(
        self._parent, self._multiprocessing_options
    )


# Keys in `MultiprocessPrefetchLazyDatasetIterator` checkpoints.
_WORKERS_STATE = "workers_state"
_ITERATIONS_TO_SKIP = "iterations_to_skip"
_LAST_WORKER_INDEX = "last_worker_index"

# Minimal interval (in seconds) between consecutive state recordings in worker
# processes of `MultiprocessPrefetchLazyDatasetIterator`. We record the state
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


class MultiprocessPrefetchLazyDatasetIterator(LazyDatasetIterator[T]):
  """Iterator that performs prefetching using a multiprocessing pool."""

  def __init__(
      self,
      parent: LazyIterDataset[T],
      multiprocessing_options: grain_options.MultiprocessingOptions,
  ):
    super().__init__()
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

  def __iter__(self) -> LazyDatasetIterator[T]:
    return self

  def __next__(self) -> T:
    if self._iterator is None:
      state = self._state
      parent = self._parent

      def get_element_producer_fn(
          worker_index: int, worker_count: int
      ) -> Iterator[tuple[T, dict[str, Any] | None]]:
        # Recover from the last recorded state for the given worker.
        worker_state = state[_WORKERS_STATE][str(worker_index)]
        parent.set_parent_maps_slice(slice(worker_index, None, worker_count))
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

      self._raw_iterator = grain_pool.MultiProcessIterator(
          get_element_producer_fn,
          self._multiprocessing_options,
          (self._state[_LAST_WORKER_INDEX] + 1)
          % self._multiprocessing_options.num_workers,
      )
      self._iterator = _iterator_with_context(self._raw_iterator)

    result, state = next(self._iterator)
    worker_index = self._raw_iterator.get_last_worker_index()  # pytype: disable=attribute-error
    self._state[_LAST_WORKER_INDEX] = worker_index
    worker_index_str = str(worker_index)
    if state is None:
      self._state[_ITERATIONS_TO_SKIP][worker_index_str] += 1
    else:
      self._state[_ITERATIONS_TO_SKIP][worker_index_str] = 0
      self._state[_WORKERS_STATE][worker_index_str] = state
    return _open_struct_from_shm(result)

  def set_state(self, state):
    self._state = state
    self._raw_iterator = None
    self._iterator = None

  def get_state(self) -> dict[str, Any]:
    return copy.deepcopy(self._state)


class RangeLazyMapDataset(LazyMapDataset[int]):
  """Range data source, similar to python range() function."""

  def __init__(self, start: int, stop: int | None = None, step: int = 1):
    super().__init__()
    self.start = 0 if stop is None else start
    self.stop = start if stop is None else stop
    self.step = step

  @functools.cached_property
  def _length(self) -> int:
    return len(range(self.start, self.stop, self.step))

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    return self.start + (index % self._length) * self.step

  def to_iter_dataset(
      self,
      read_options: grain_options.ReadOptions | None = None,
  ) -> LazyIterDataset[int]:
    """Syntactic sugar to construct a LazyIterDataset."""
    return PrefetchLazyIterDataset(
        self,
        read_options=(
            read_options or grain_options.ReadOptions(prefetch_buffer_size=0)
        ),
    )


# Deprecated: This class should not be used for new code. It's used to
# implement the stateless Sampler.
# For new code the PrefetchLazyMapDataset should be used to implement sharding.
class ShardLazyDataset(LazyMapDataset[T]):
  """Shards the parent into consecutive pieces."""

  def __init__(
      self, parent: LazyMapDataset[T], shard_options: sharding.ShardOptions
  ):
    super().__init__(parent)
    self._start, self._end = sharding.even_split(
        len(self._parent), shard_options
    )

  def __len__(self) -> int:
    return self._end - self._start

  def __getitem__(self, index: int | slice) -> Optional[T]:
    if isinstance(index, slice):
      return self.slice(index)
    epoch = index // len(self)
    index_in_epoch = index % len(self)
    index = epoch * len(self._parent) + index_in_epoch + self._start
    return self._parent[index]
