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
from typing import Any, Optional, TypeVar, Union, overload

from grain._src.core import monitoring as grain_monitoring
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import options as grain_options
from grain._src.python.lazy_dataset import base
import numpy as np

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
S = TypeVar("S")


class _SeededDataset(abc.ABC):
  """Node of a dataset tree structure that supports generating default seed."""

  def __init__(self):
    # Seeds a `SeedSequence` used to generate default seeds for all
    # downstream transformations. Set by `_WithOptions{Map|Iter}Dataset`.
    self._seed_rng_seed = None

  @property
  @abc.abstractmethod
  def parents(self) -> Sequence[Any]:
    """Returns the parent nodes of this dataset."""
    ...

  @functools.cached_property
  def _default_seed(self) -> int | None:
    """Should be used as a seed if no seed is provided."""
    aggregated_seed = []
    # Note that the traversal order must be determisitic.
    to_visit = [(self, 0)]
    while to_visit:
      node, depth = to_visit.pop(0)
      if (node_seed := node._seed_rng_seed) is not None:  # pylint: disable=protected-access
        aggregated_seed.extend((node_seed, depth))
      else:
        to_visit.extend((n, depth + 1) for n in node.parents)
    if not aggregated_seed:
      return None
    seed_sequence = np.random.SeedSequence(aggregated_seed)
    return seed_sequence.generate_state(1, dtype=np.uint32)[0]


class _MapDatasetMeta(abc.ABCMeta):
  """Metaclass for MapDataset containing factory transfromations."""

  def source(cls, source: base.RandomAccessDataSource[T]) -> MapDataset[T]:
    """Returns a dataset that wraps a data source supporting random access.

    Example usage: `ds = MapDataset.source(ArrayRecordDataSource(paths))`.

    Args:
      source: Data source supporting efficient random access.

    Returns:
      A MapDataset that wraps the data source and allows to chain other
      MapDataset transformations.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> source).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import source as source_dataset
    # pylint: enable=g-import-not-at-top
    return source_dataset.SourceMapDataset(source)

  def range(
      cls, start: int, stop: int | None = None, step: int = 1
  ) -> MapDataset[int]:
    """Returns a dataset with a range of integers.

    Input arguments are interpreted the same way as in Python built-in `range`:
      - `range(n)` => start=0, stop=n, step=1
      - `range(m, n)` => start=m, stop=n, step=1
      - `range(m, n, p)` => start=m, stop=n, step=p

    `list(MapDataset.range(...)) == list(range(...))`.

    Args:
      start: The start of the range.
      stop: The stop of the range.
      step: The step of the range.

    Returns:
      A MapDataset with a range of integers.
    """
    return RangeMapDataset(start, stop, step)

  def mix(
      cls,
      datasets: Sequence[MapDataset[T]],
      weights: Sequence[float] | None = None,
  ) -> MapDataset[T]:
    """Returns a dataset that mixes input datasets with the given weights.

    Length of the mixed dataset will be determined by the length of the shortest
    input dataset. If you need an infinite dateset consider repeating the
    input datasets before mixing.

    If you need to shuffle the mixed dataset while preserving the correct
    proportions, you should shuffle the input datasets before mixing.

    Args:
      datasets: The datasets to mix.
      weights: The weights to use for mixing. Defaults to uniform weights if not
        specified.

    Returns:
      A MapDataset that represents a mixture of the input datasets according
      to the given weights.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedMapDataset(parents=datasets, proportions=weights)

  def select_from_datasets(
      cls,
      datasets: Sequence[MapDataset[T]],
      selection_map: base.DatasetSelectionMap,
  ) -> MapDataset[T]:
    """Returns a dataset selected from the inputs accoridng to the given map.

    Allows more general types of dataset mixing than `mix`.

    Args:
      datasets: The datasets to select from.
      selection_map: Mapping from index within the mixed dataset to a selected
        dataset index and index within that dataset. Length of the resulting
        dataset will be determined by the length of the `selection_map`.

    Returns:
      A MapDataset that represents a mixture of the input datasets according
      to the given selection map.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedMapDataset(parents=datasets, selection_map=selection_map)


class MapDataset(_SeededDataset, Sequence[T], metaclass=_MapDatasetMeta):
  """Abstract base class for all MapDataset classes."""

  def __init__(self, parents: Union[MapDataset, Sequence[MapDataset]] = ()):
    super().__init__()
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

  def batch(
      self,
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> "MapDataset[S]":
    """Returns a dataset of elements batched along a new first dimension.

    Dataset elements are expected to be PyTrees.

    Args:
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new first batch dimension.

    Returns:
      A dataset of elements with the same PyTree structure with leaves
      concatenated along the first dimension.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> batch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import batch
    # pylint: enable=g-import-not-at-top
    return batch.BatchMapDataset(
        parent=self,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        batch_fn=batch_fn,
    )

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

  def map(
      self,
      transform: (
          transforms.MapTransform
          | Callable[[T], S]
          | transforms.RandomMapTransform
          | Callable[[T, np.random.Generator], S]
      ),
      seed: Optional[int] = None,
  ) -> "MapDataset[S]":
    """Returns a dataset containing the elements transformed by `transform`.

    The following expressions are equivalent:

    - `ds = ds.map(lambda x: x + 1)`
    - `ds = MapMapDataset(ds, lambda x: x + 1)`

    The `ds.map(...)` version allows chaining multiple transformations,
    e.g., `ds = ds.map(...).filter(...)`.

    Args:
      transform: Either a `MapTransform` containing the `map` method or a
        callable that takes an element and returns a new element. The
        `RandomMapTransform` and `Callable[[T, np.random.Generator], S]` types
        are deprecated (use `random_map` instead).
      seed: Deprecated. Use `random_map` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import map as map_dataset
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapMapDataset(
        parent=self, transform=transform, seed=seed
    )

  def map_with_index(
      self,
      transform: transforms.MapWithIndexTransform | Callable[[int, T], S],
  ) -> "MapDataset[S]":
    """Returns a dataset containing the elements transformed by `transform`.

    The following expressions are equivalent:

    - `ds = ds.map_with_index(lambda i, x: i + 2 * x)`
    - `ds = MapWithIndexMapDataset(ds, lambda i, x: i + 2 * x)`

    The `ds.map_with_index(...)` version allows chaining multiple
    transformations, e.g., `ds = ds.map_with_index(...).filter(...)`.

    Args:
      transform: Either a `MapWithIndexTransform` containing the
        `map_with_index` method or a callable that takes an index and an element
        and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import map as map_dataset
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapWithIndexMapDataset(parent=self, transform=transform)

  def seed(self, seed: int) -> MapDataset[T]:
    """Returns a dataset that uses the seed for default seed generation.

    When default seed generation is enabled by calling `ds.seed`, every
    downstream random transformation will be automatically seeded with a unique
    seed by default. This simplifies seed management, making it easier to avoid:
     - Having to provide a seed in multiple transformations.
     - Accidentally reusing the same seed across transformations.

    It is recommended to call this right after the source. `ds.seed` has to be
    called before any random transformations (such as `shuffle` or `random_map`
    that rely on default seed generation to control their seeding). Given the
    same seed, the pipeline is guaranteed to always use the same seeds for each
    transformation.

    Note about custom dataset implementations: the default seed generation is
    available through `_default_seed`, but the private API is not guaranteed to
    be stable.

    Example usage:
    `ds = ds.seed(seed).shuffle()`.
    `shuffle` will automatically derive its own seed (different from `seed`).

    `ds = ds.seed(seed).shuffle().random_map(...)`.
    `shuffle` and `random_map` will each derive their own seed and the seeds are
    going to be different.

    `ds = ds.seed(seed).random_map(transform, seed=seed1)`.
    `random_map` will use `seed1` and will not be affected by `seed`. This
    can be used to control individual transformation seeding independently from
    the rest of the pipeline.

    `ds = ds.seed(seed1).shuffle().seed(seed2).random_map(...)`.
    `ds.seed` only affects the downstream transformations and can be overridden
    by a subsequent `seed` call.
    `shuffle` will derive its seed from `seed1`, `random_map` - from `seed2` and
    will not be affected by `seed1`. This can be used to control your
    transformation seeding even if you don't own the first part of the pipeline.

    ```
    ds1 = ds.source(...).seed(seed1).shuffle()
    ds2 = ds.source(...).seed(seed2).shuffle()
    ds = MapDataset.mix([ds1, ds2], ...).random_map(...)
    ```
    Each `shuffle` will derive its own seed from `seed1` or `seed2`
    respectively. `random_map` will derive its seed from both `seed1` and
    `seed2`.

    Args:
      seed: Seed to use.

    Returns:
      A dataset with elements unchanged.
    """
    return _WithOptionsMapDataset(parent=self, seed=seed)

  def shuffle(self, *, seed: int | None = None) -> "MapDataset[T]":
    """Returns a dataset containing the same elements but in a shuffled order.

    The following expressions are equivalent:

    - `ds = ds.shuffle(seed=42)`
    - `ds = ShuffleMapDataset(ds, seed=42)`

    The `ds.shuffle(...)` version allows chaining multiple transformations,
    e.g.,
    `ds = ds.filter(...).map(...).shuffle(...)`.

    Args:
      seed: An optional integer between 0 and 2**32-1 representing the seed used
        by the shuffling algorithm. If you don't need to control the shuffle
        seed individually, prefer setting the pipeline-level seed with
        `ds.seed(seed)` instead.

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

  def random_map(
      self,
      transform: (
          transforms.RandomMapTransform | Callable[[T, np.random.Generator], S]
      ),
      *,
      seed: int | None = None,
  ) -> "MapDataset[S]":
    """Returns a dataset containing the elements transformed by `transform`.

    The following expressions are equivalent:

    - `ds = ds.random_map(lambda x, rng: x + rng.integers(0, 100), seed=42)`
    - `ds = MapMapDataset(ds, lambda x, rng: x + rng.integers(0, 100),
        seed=42)`

    The `ds.random_map(...)` version allows chaining multiple transformations,
    e.g., `ds = ds.random_map(...).filter(...)`.

    Args:
      transform: Either a `RandomMapTransform` containing the `random_map`
        method or a callable that takes an element and a np.random.Generator and
        returns a new element.
      seed: An optional integer between 0 and 2**32-1 representing the seed used
        to initialize the random number generator used by `transform`. If you
        don't need to control the shuffle seed individually, prefer setting the
        pipeline-level seed with`ds.seed(seed)` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import map as map_dataset
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapMapDataset(
        parent=self, transform=transform, seed=seed
    )

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


class _IterDatasetMeta(abc.ABCMeta):
  """Metaclass for IterDataset containing factory transformations."""

  def mix(
      cls,
      datasets: Sequence[IterDataset[T]],
      weights: Sequence[float] | None = None,
  ) -> IterDataset[T]:
    """Returns a dataset that mixes input datasets with the given weights.

    Length of the mixed dataset will be determined by the length of the shortest
    input dataset. If you need an infinite mixed dateset consider repeating the
    input datasets before mixing.

    Args:
      datasets: The datasets to mix.
      weights: The weights to use for mixing. Defaults to uniform weights if not
        specified.

    Returns:
      A dataset that represents a mixture of the input datasets according to the
      given weights.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedIterDataset(parents=datasets, proportions=weights)


class IterDataset(_SeededDataset, Iterable[T], metaclass=_IterDatasetMeta):
  """Abstract base class for all IterDataset classes."""

  def __init__(
      self,
      parents: Union[
          MapDataset,
          IterDataset,
          Sequence[Union[MapDataset, IterDataset]],
      ] = (),
  ):
    super().__init__()
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

  def batch(
      self,
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> "IterDataset[S]":
    """Returns a dataset of elements batched along a new first dimension.

    Dataset elements are expected to be PyTrees.

    Args:
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new first batch dimension.

    Returns:
      A dataset of elements with the same PyTree structure with leaves
      concatenated along the first dimension.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> batch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import batch
    # pylint: enable=g-import-not-at-top
    return batch.BatchIterDataset(
        parent=self,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        batch_fn=batch_fn,
    )

  def seed(self, seed: int) -> IterDataset[T]:
    """Returns a dataset that uses the seed for default seed generation.

    When default seed generation is enabled by calling `ds.seed`, every
    downstream random transformation will be automatically seeded with a unique
    seed by default. This simplifies seed management, making it easier to avoid:
     - Having to provide a seed in multiple transformations.
     - Accidentally reusing the same seed across transformations.

    It is recommended to call this right after the source. `ds.seed` has to be
    called before any random transformations (such as `random_map` that rely on
    default seed generation to control their seeding). Given the same seed, the
    pipeline is guaranteed to always use the same seeds for each transformation.

    Note about custom dataset implementations: the default seed generation is
    available through `_default_seed`, but the private API is not guaranteed to
    be stable.

    Example usage:
    `ds = ds.seed(seed).random_map(...)`.
    `random_map` will automatically derive its own seed (different from `seed`).

    `ds = ds.seed(seed).random_map().random_map(...)`.
    The first and second `random_map`s will each derive their own seed and the
    seeds are going to be different.

    `ds = ds.seed(seed).random_map(transform, seed=seed1)`.
    `random_map` will use `seed1` and will not be affected by `seed`. This
    can be used to control individual transformation seeding independently from
    the rest of the pipeline.

    `ds = ds.seed(seed1).random_map(...).seed(seed2).random_map(...)`.
    `ds.seed` only affects the downstream transformations and can be overridden
    by a subsequent `seed` call.
    The first `random_map` will derive its seed from `seed1`, the second - from
    `seed2` and will not be affected by `seed1`. This can be used to control
    your
    transformation seeding even if you don't own the first part of the pipeline.

    ```
    ds1 = ds.source(...).seed(seed2).shuffle().to_iter_dataset()
    ds2 = ds.source(...).seed(seed2).shuffle().to_iter_dataset()
    ds = IterDataset.mix([ds1, ds2], ...).random_map(...)
    ```
    Each `shuffle` will derive its own seed from `seed1` or `seed2`
    respectively. `random_map` will derive its seed from both `seed1` and
    `seed2`.

    Args:
      seed: Seed to use.

    Returns:
      A dataset with elements unchanged.
    """
    return _WithOptionsIterDataset(parent=self, seed=seed)

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

  def map(
      self,
      transform: (
          transforms.MapTransform
          | Callable[[T], S]
          | transforms.RandomMapTransform
          | Callable[[T, np.random.Generator], S]
      ),
      seed: Optional[int] = None,
  ) -> "IterDataset[S]":
    """Returns a dataset containing the elements transformed by `transform`.

    The following expressions are equivalent:

    - `ds = ds.map(lambda x: x + 1)`
    - `ds = MapIterDataset(ds, lambda x: x + 1)`

    The `ds.map(...)` version allows chaining multiple transformations,
    e.g., `ds = ds.map(...).filter(...)`.

    Args:
      transform: Either a `MapTransform` containing the `map` method or a
        callable that takes an element and returns a new element. The
        `RandomMapTransform` and `Callable[[T, np.random.Generator], S]` types
        are deprecated (use `random_map` instead).
      seed: Deprecated. Use `random_map` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import map as map_dataset
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapIterDataset(
        parent=self, transform=transform, seed=seed
    )

  def random_map(
      self,
      transform: (
          transforms.RandomMapTransform | Callable[[T, np.random.Generator], S]
      ),
      *,
      seed: int | None = None,
  ) -> "IterDataset[S]":
    """Returns a dataset containing the elements transformed by `transform`.

    The following expressions are equivalent:

    - `ds = ds.random_map(lambda x, rng: x + rng.integers(0, 100), seed=42)`
    - `ds = MapIterDataset(ds, lambda x, rng: x + rng.integers(0, 100),
        seed=42)`

    The `ds.random_map(...)` version allows chaining multiple transformations,
    e.g., `ds = ds.random_map(...).filter(...)`.

    Args:
      transform: Either a `RandomMapTransform` containing the `random_map`
        method or a callable that takes an element and a np.random.Generator and
        returns a new element.
      seed: An integer between 0 and 2**32-1 representing the seed used to
        initialize the random number generator used by `transform`. If you don't
        need to control the transformation seed individually, prefer setting the
        pipeline-level seed with`ds.seed(seed)` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (lazy_dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.lazy_dataset.transformations import map as map_dataset
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapIterDataset(
        parent=self, transform=transform, seed=seed
    )

  def _set_parent_maps_slice(self, sl: slice) -> None:
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
        parent._set_parent_maps_slice(sl)  # pylint: disable=protected-access
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


class _WithOptionsMapDataset(MapDataset[T]):
  """Holds options used by downstream transformations."""

  def __init__(self, parent: MapDataset[T], *, seed: int):
    super().__init__(parent)
    self._seed_rng_seed = seed

  def __len__(self) -> int:
    return self._parent.__len__()

  def __getitem__(self, index) -> T:
    return self._parent[index]


class _WithOptionsIterDataset(IterDataset[T]):
  """Holds options used by downstream transformations."""

  def __init__(self, parent: IterDataset[T], *, seed: int):
    super().__init__(parent)
    self._seed_rng_seed = seed

  def __iter__(self) -> DatasetIterator[T]:
    return self._parent.__iter__()


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
