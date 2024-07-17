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
- `MapDataset` define a dataset that supports efficient random access. It
  has 3 important properties:
  - `__len__()` returns the length of a single epoch over the dataset.
  - `__getitem__()` will return the element at any given (positive) index. The
    "true" length of a `MapDataset` is infinite. Many implementations will
    simply loop but exceptions exists (e.g. `ShuffleMapDataset` will loop
    with a different order).
  - The dataset is lazy and individual elements are only created when calling
    `__getitem__()`. Most `MapDatasets`s are statements and will not hold
    elements.
- `IterDataset` defines a dataset that does not support efficient random
  access. It can still be iterated over. A `MapDataset` can be turned into
  a `IterDataset` but going from `IterDataset` to `MapDataset` might
  be as expensive as materializing the whole dataset.
  A `IterDataset` can have known, unknown or infinite length.
- `DatasetIterator` defines a stateful iterator over `IterDataset`. The
  state of the iterator can be saved and restored.

Using the interfaces defined in `collections.abc` you can think of
MapDataset as (infinite) Sequence, IterDataset as Iterable and
DatasetIterator as Iterator.
"""

from __future__ import annotations

import abc
import builtins
from collections.abc import Callable, Iterable, Iterator, Sequence
import functools
from typing import Any, Optional, Protocol, TypeVar, Union, overload

from grain._src.core import monitoring as grain_monitoring
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import options as grain_options

from grain._src.core import monitoring


_api_usage_counter = monitoring.Counter(
    "/grain/python/lazy_dataset/api",
    metadata=monitoring.Metadata(
        description="Lazy Dataset API initialization counter."
    ),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)

T = TypeVar("T")
_MAX_PREFETCH_THREADS = 1000


class RegisterableMapDatasetFn(Protocol):
  """Interface for functions registered on all MapDatasets."""

  def __call__(self, dataset: MapDataset, *args, **kwargs) -> Any:
    ...


class RegisterableIterDatasetFn(Protocol):
  """Interface for functions registered on all IterDatasets."""

  def __call__(self, dataset: IterDataset, *args, **kwargs) -> Any:
    ...


class MapDataset(Sequence[T], abc.ABC):
  """Abstract base class for all MapDataset classes."""

  _functions: dict[str, RegisterableMapDatasetFn] = {}
  """Functions registered on all MapDatasets via a decoration."""

  def __init__(self, parents: Union[MapDataset, Sequence[MapDataset]] = ()):
    if isinstance(parents, MapDataset):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    usage_logging.log_event("MapDataset", tag_3="PyGrain")
    _api_usage_counter.Increment("MapDataset")

  @property
  def parents(self) -> Sequence[MapDataset]:
    return self._parents

  @property
  def _parent(self) -> MapDataset:
    assert len(self._parents) == 1
    return self._parents[0]

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the length of this dataset."""

  @overload
  def __getitem__(self, index: builtins.slice) -> MapDataset:
    ...

  @overload
  def __getitem__(self, index: int) -> Optional[T]:
    ...

  @abc.abstractmethod
  def __getitem__(self, index):
    """Returns the element for the index or None if missing."""

  def filter(
      self, transform: transforms.FilterTransform | Callable[[T], bool]
  ) -> "MapDataset[T]":
    """Returns a dataset containing only the elements that match the filter.

    Accessing an element of the returned dataset using subscription (`ds[i]`)
    returns:

    - `None` if `transform` returned `False`
    - the element if `transform` returned `True`

    Iterating over a filtered dataset skips `None` elements by default.

    The following expressions are equivalent:

    - `ds = ds.filter(lambda x: x > 5)`
    - `ds = FilterMapDataset(ds, lambda x: x > 5)`

    The `ds.filter(...)` version allows chaining multiple transformations, e.g.,
    `ds = ds.filter(...).map(...).filter(...)`

    Args:
      transform: Either a `FilterTransform` containing the `filter` method or a
        callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns `True`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> filter).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import filter as filter_dataset
    # pylint: enable=g-import-not-at-top
    return filter_dataset.FilterMapDataset(parent=self, transform=transform)

  def shuffle(self, *, seed: int) -> "MapDataset[T]":
    """Returns a dataset containing the same elements but in a shuffled order.

    The following expressions are equivalent:

    - `ds = ds.shuffle(seed=42)`
    - `ds = ShuffleMapDataset(ds, seed=42)`

    The `ds.shuffle(...)` version allows chaining multiple transformations,
    e.g.,
    `ds = ds.filter(...).map(...).shuffle(...)`.

    Args:
      seed: An integer between 0 and 2**32-1 representing the seed used by the
        shuffling algorithm.

    Returns:
      A dataset containing the same elements but in a shuffled order.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> shuffle).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import shuffle
    # pylint: enable=g-import-not-at-top
    return shuffle.ShuffleMapDataset(parent=self, seed=seed)

  def slice(self, sl: builtins.slice) -> "MapDataset[T]":
    """Returns a dataset containing only the elements with indices in `sl`.

    The following expressions are equivalent:

    - `ds = ds.slice(slice(1, 10, 2))`
    - `ds = SliceMapDataset(ds, slice(1, 10, 2))`
    - `ds = ds[1:10:2]` (for `MapDataset`s supporting `slice` objects in
      subscriptions)

    The `ds.slice(...)` and `ds[...]` versions allow chaining multiple
    transformations, e.g.,
    `ds = ds[10::4].filter(...).map(...)`.

    Args:
      sl: A `slice` object
        (https://docs.python.org/3/library/functions.html#slice) representing
        the slice of elements to that should constitute the returned dataset.

    Returns:
      A dataset containing only the elements with indices in the `sl` slice.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> slice).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import slice as slice_dataset
    # pylint: enable=g-import-not-at-top
    return slice_dataset.SliceMapDataset(parent=self, sl=sl)

  def repeat(self, num_epochs: int | None = None) -> "MapDataset[T]":
    """Returns a dataset repeating the elements of this dataset multiple times.

    Specifying `None` for `num_epochs` will repeat the dataset infinitely, and
    causes `len(ds)` to return `sys.maxsize`.

    Since `MapDataset`s allow accessing elements past `len(ds) - 1` anyway
    (and use the index modulo `len(ds)`), this transformation effectively only
    changes the length of the dataset.

    `repeat(...)` shouldn't be called on an infinite dataset.

    The following expressions are equivalent:

    - `ds = ds.repeat(42)`
    - `ds = RepeatMapDataset(ds, 42)`

    The `ds.repeat(...)` version allows chaining multiple transformations, e.g.,
    `ds = ds.filter(...).map(...).repeat(...)`.

    Args:
      num_epochs: Either a positive integer representing the number of times
        this dataset should be repeated or `None` to repeat infinitely.

    Returns:
      A dataset repeating the elements of this dataset multiple times.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> repeat).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import repeat
    # pylint: enable=g-import-not-at-top
    return repeat.RepeatMapDataset(parent=self, num_epochs=num_epochs)

  @classmethod
  def register_function(cls, name: str, function: RegisterableMapDatasetFn):
    if name in cls._functions:
      raise ValueError(
          f"Cannot register {function} as dataset function '{name}' since it's"
          f" already taken by {cls._functions[name]}."
      )
    cls._functions[name] = function

  def __getattr__(self, attribute_name: str):
    if attribute_name in MapDataset._functions:
      return functools.partial(MapDataset._functions[attribute_name], self)
    raise AttributeError(
        f"'{self.__class__.__name__}' object has no attribute"
        f" '{attribute_name}' :("
    )

  def __iter__(self) -> DatasetIterator[T]:
    return self.to_iter_dataset().__iter__()

  def to_iter_dataset(
      self,
      read_options: Optional[grain_options.ReadOptions] = None,
      allow_nones: bool = False,
  ) -> IterDataset[T]:
    """Syntactic sugar to construct a IterDataset."""
    # Loaded lazily due to a circular dependency (lazy_dataset <-> prefetch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import prefetch
    # pylint: enable=g-import-not-at-top
    return prefetch.PrefetchIterDataset(
        self,
        read_options=read_options or grain_options.ReadOptions(),
        allow_nones=allow_nones,
    )


class IterDataset(Iterable[T], abc.ABC):
  """Abstract base class for all IterDataset classes."""

  _functions: dict[str, RegisterableIterDatasetFn] = {}

  def __init__(
      self,
      parents: Union[
          MapDataset,
          IterDataset,
          Sequence[Union[MapDataset, IterDataset]],
      ] = (),
  ):
    if isinstance(parents, (MapDataset, IterDataset)):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    usage_logging.log_event("IterDataset", tag_3="PyGrain")
    _api_usage_counter.Increment("IterDataset")

  @property
  def parents(self) -> Sequence[Union[MapDataset, IterDataset]]:
    return self._parents

  @property
  def _parent(self) -> Union[MapDataset, IterDataset]:
    assert len(self._parents) == 1, self._parents
    return self._parents[0]

  def filter(
      self, transform: transforms.FilterTransform | Callable[[T], bool]
  ) -> "IterDataset[T]":
    """Returns a dataset containing only the elements that match the filter.

    `ds = ds.filter(lambda x: x > 5)`
    is equivalent to
    `ds = FilterIterDataset(ds, lambda x: x > 5)`

    Args:
      transform: Either a `FilterTransform` containing the `filter` method or a
        callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns `True`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> filter).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import filter as filter_dataset
    # pylint: enable=g-import-not-at-top
    return filter_dataset.FilterIterDataset(parent=self, transform=transform)

  def set_parent_maps_slice(self, sl: slice) -> None:
    """Replaces MapDataset-type parents with their sliced versions.

    Applies recursively for IterDataset-type parents.

    Args:
     sl: slice to apply.
    """
    sliced_parents = []
    for parent in self._parents:
      if isinstance(parent, MapDataset):
        sliced_parents.append(parent.slice(sl))
      else:
        parent.set_parent_maps_slice(sl)
        sliced_parents.append(parent)
    self._parents = tuple(sliced_parents)

  def prefetch(
      self, multiprocessing_options: grain_options.MultiprocessingOptions
  ) -> "IterDataset[T]":
    """Returns a dataset prefetching the elements in multiple processes.

    Each of the processes will process a slice of the dataset after all
    MapDataset transformations.

    WARNING: If the dataset contains many-to-one transformations (such as
    `batch`), output after prefetch may change if you change the number of
    workers. However, it is still going to be determisitic.

    Args:
      multiprocessing_options: options for the prefetching processes.
        `num_workers` must be greater than 0.

    Returns:
      A dataset prefetching input elements concurrently.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> prefetch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import prefetch
    # pylint: enable=g-import-not-at-top
    return prefetch.MultiprocessPrefetchIterDataset(
        self, multiprocessing_options=multiprocessing_options
    )

  @abc.abstractmethod
  def __iter__(self) -> DatasetIterator[T]:
    """Returns an iterator for this dataset."""

  @classmethod
  def register_function(cls, name: str, function: RegisterableMapDatasetFn):
    if name in cls._functions:
      raise ValueError(
          f"Cannot register {function} as dataset function '{name}' since it's"
          f" already taken by {cls._functions[name]}."
      )
    cls._functions[name] = function

  def __getattr__(self, attribute_name: str):
    if attribute_name in IterDataset._functions:
      return functools.partial(IterDataset._functions[attribute_name], self)
    raise AttributeError(
        f"'{self.__class__.__name__}' object has no attribute"
        f" '{attribute_name}' :("
    )


def lazy_map_dataset_function(name: str):
  """Registers a function as a MapDataset function."""

  def _fn(cls):
    MapDataset.register_function(name=name, function=cls)
    return cls

  return _fn


def lazy_iter_dataset_function(name: str):
  """Registers a function as a IterDataset function."""

  def _fn(cls):
    IterDataset.register_function(name=name, function=cls)
    return cls

  return _fn


class DatasetIterator(Iterator[T], abc.ABC):
  """Abstract base class for all IterDataset iterator classes."""

  def __iter__(self) -> DatasetIterator[T]:
    return self

  # __next__ abstract method since we inherit from Iterator[T].

  @abc.abstractmethod
  def get_state(self) -> dict[str, Any]:
    """Returns the current state of the iterator."""

  @abc.abstractmethod
  def set_state(self, state: dict[str, Any]):
    """Sets the current state of the iterator."""


class RangeMapDataset(MapDataset[int]):
  """Range data source, similar to python range() function."""

  def __init__(self, start: int, stop: Optional[int] = None, step: int = 1):
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
      read_options: Optional[grain_options.ReadOptions] = None,
      allow_nones: bool = False,
  ) -> IterDataset[int]:
    """Syntactic sugar to construct a IterDataset."""
    return super().to_iter_dataset(
        read_options=(
            read_options or grain_options.ReadOptions(prefetch_buffer_size=0)
        ),
        allow_nones=allow_nones,
    )


# Deprecated: This class should not be used for new code. It's used to
# implement the stateless Sampler.
# For new code the PrefetchMapDataset should be used to implement sharding.
class ShardLazyDataset(MapDataset[T]):
  """Shards the parent into consecutive pieces."""

  def __init__(
      self, parent: MapDataset[T], shard_options: sharding.ShardOptions
  ):
    super().__init__(parent)
    self._start, self._end = sharding.even_split(
        len(self._parent), shard_options
    )

  def __len__(self) -> int:
    return self._end - self._start

  def __getitem__(self, index: Union[int, slice]) -> Optional[T]:
    if isinstance(index, slice):
      return self.slice(index)
    epoch = index // len(self)
    index_in_epoch = index % len(self)
    index = epoch * len(self._parent) + index_in_epoch + self._start
    return self._parent[index]
