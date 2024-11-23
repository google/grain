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
  - The individual dataset elements are only evaluated when calling
    `__getitem__()`. `MapDatasets`s are stateless and will not hold
    elements.
- `IterDataset` defines a dataset that does not support efficient random
  access. It can still be iterated over. A `MapDataset` can be turned into
  a `IterDataset` but going from `IterDataset` to `MapDataset` might
  be as expensive as materializing the whole dataset.
- `DatasetIterator` defines a stateful iterator over `IterDataset`. The
  state of the iterator can be saved and restored.

Using the interfaces defined in `collections.abc` you can think of
MapDataset as (infinite) Sequence, IterDataset as Iterable and
DatasetIterator as Iterator.
"""

from __future__ import annotations

import abc
import builtins
import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings

from grain._src.core import monitoring as grain_monitoring
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import stats as dataset_stats
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


class _Dataset:
  """Node of a dataset tree structure that represents data transfromation.

  Supports generating default seed for random transformations.
  """

  # Whether this transformation mutates parent elements. This does not affect
  # the transformation itself, only used for information purposes in statistics.
  _MUTATES_ELEMENT_SPEC = True

  def __init__(self, parents: Sequence[_Dataset]):
    # Seeds a `SeedSequence` used to generate default seeds for all
    # downstream transformations. Set by `_WithOptions{Map|Iter}Dataset`.
    self._seed_rng_seed = None
    self._parents = parents

  @functools.cached_property
  def _default_seed(self) -> int | None:
    """Should be used as a seed if no seed is provided."""
    aggregated_seed = []
    # Note that the traversal order must be determisitic.
    # pylint:  disable=protected-access
    to_visit = [(self, 0)]
    while to_visit:
      node, depth = to_visit.pop(0)
      if (node_seed := getattr(node, "_seed_rng_seed", None)) is not None:
        aggregated_seed.extend((node_seed, depth))
      else:
        to_visit.extend((n, depth + 1) for n in node._parents)
    # pylint:  enable=protected-access
    if not aggregated_seed:
      return None
    seed_sequence = np.random.SeedSequence(aggregated_seed)
    return seed_sequence.generate_state(1, dtype=np.uint32)[0]


class _MapDatasetMeta(abc.ABCMeta):
  """Metaclass for `MapDataset` containing factory transfromations."""

  def source(cls, source: base.RandomAccessDataSource[T]) -> MapDataset[T]:
    """Returns a dataset that wraps a data source supporting random access.

    Example usage: `ds = MapDataset.source(ArrayRecordDataSource(paths))`.

    Works with Sequence inputs as well:
    `list(MapDataset.source([1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]`.

    Args:
      source: Data source supporting efficient random access.

    Returns:
      A MapDataset that wraps the data source and allows to chain other
      MapDataset transformations.
    """
    # Loaded lazily due to a circular dependency (dataset <-> source).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        source as source_dataset,
    )
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
    # Loaded lazily due to a circular dependency (dataset <-> source).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        source as source_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return source_dataset.RangeMapDataset(start, stop, step)

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

    Example usage:
    ```
    ds1 = MapDataset.range(5)
    ds2 = MapDataset.range(7, 10)
    list(MapDataset.mix([ds1, ds2])) == [0, 7, 1, 8, 2, 9]
    ```

    Args:
      datasets: The datasets to mix.
      weights: The weights to use for mixing. Defaults to uniform weights if not
        specified.

    Returns:
      A MapDataset that represents a mixture of the input datasets according
      to the given weights.
    """
    # Loaded lazily due to a circular dependency (dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import mix
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
    # Loaded lazily due to a circular dependency (dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedMapDataset(parents=datasets, selection_map=selection_map)


class MapDataset(_Dataset, Generic[T], metaclass=_MapDatasetMeta):
  """Represents a dataset with transformations that support random access.

  Transformations do not mutate the dataset object. Instead, they return a new
  dataset. From the perspective of public APIs, `MapDataset` is immutable.
  """

  # Whether this transformation mutates parent elements. This does not affect
  # the transformation itself, only used for information purposes in statistics.
  _MUTATES_ELEMENT_SPEC = True

  def __init__(self, parents: MapDataset | Sequence[MapDataset] = ()):
    if isinstance(parents, MapDataset):
      parents = (parents,)
    else:
      parents = tuple(parents)
    super().__init__(parents)
    self._parents = cast(Sequence[MapDataset], self._parents)
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
    """Returns length of a single epoch of this dataset."""

  @overload
  def __getitem__(self, index: builtins.slice) -> MapDataset[T]:
    ...

  @overload
  def __getitem__(self, index: int) -> T | None:
    ...

  @abc.abstractmethod
  def __getitem__(self, index):
    """Returns the element for the index or None if missing."""

  def batch(
      self,
      batch_size: int,
      *,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> MapDataset[S]:
    """Returns a dataset of elements batched along a new first dimension.

    Dataset elements are expected to be PyTrees.

    Example usage:
    ```
    ds = MapDataset.range(5)
    ds = ds.batch(batch_size=2)
    list(ds) == [np.ndarray([0, 1]), np.ndarray([2, 3]), np.ndarray([4])]
    ```

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
    # Loaded lazily due to a circular dependency (dataset <-> batch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import batch
    # pylint: enable=g-import-not-at-top
    return batch.BatchMapDataset(
        parent=self,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
        batch_fn=batch_fn,
    )

  def filter(
      self, transform: transforms.FilterTransform | Callable[[T], bool]
  ) -> MapDataset[T]:
    """Returns a dataset containing only the elements that match the filter.

    Accessing an element of the returned dataset using subscription (`ds[i]`)
    returns:

    - `None` if `transform` returned `False`
    - the element if `transform` returned `True`

    Iterating over a filtered dataset skips `None` elements by default.

    Example usage:
    ```
    ds = MapDataset.range(5)
    ds = ds.filter(lambda x: x % 2 == 0)
    ds[2] == 2
    ds[1] == None
    list(ds) == [0, 2, 4]
    ```
    NOTE: `list(ds)` converts the dataset to an `IterDataset` with
    `to_iter_dataset()` under the hood which by default skips `None` elements.

    `to_iter_dataset` produces a warning when iterating through a filtered
    dataset if the filter removes more than 90% of the elements.
    You can adjust the threshold through
    `grain.experimental.DatasetOptions.filter_warn_threshold_ratio` used in
    `WithOptionsIterDataset`. In order to produce an exception in such case use
    `filter_raise_threshold_ratio`.

    Args:
      transform: Either a `FilterTransform` containing the `filter` method or a
        callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns `True`.
    """
    # Loaded lazily due to a circular dependency (dataset <-> filter).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        filter as filter_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return filter_dataset.FilterMapDataset(parent=self, transform=transform)

  def map(
      self, transform: transforms.MapTransform | Callable[[T], S]
  ) -> MapDataset[S]:
    """Returns a dataset containing the elements transformed by `transform`.

    Example usage:
    ```
    ds = MapDataset.range(5)
    ds = ds.map(lambda x: x + 10)
    list(ds) == [10, 11, 12, 13, 14]
    ```

    Args:
      transform: Either a `MapTransform` containing the `map` method or a
        callable that takes an element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapMapDataset(parent=self, transform=transform)

  def map_with_index(
      self,
      transform: transforms.MapWithIndexTransform | Callable[[int, T], S],
  ) -> MapDataset[S]:
    """Returns a dataset containing the elements transformed by `transform`.

    The transform is called with the index of the element within the dataset
    and the element itself.

    Example usage:
    ```
    ds = MapDataset.source(["a", "b", "c", "d"])
    ds = ds.map_with_index(lambda i, x: x + str(i))
    list(ds) == ["a0", "b1", "c2", "d3"]
    ```

    Args:
      transform: Either a `MapWithIndexTransform` containing the
        `map_with_index` method or a callable that takes an index and an element
        and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
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
    return _WithSeedMapDataset(parent=self, seed=seed)

  def shuffle(self, seed: int | None = None) -> MapDataset[T]:
    """Returns a dataset with the same elements in a globally shuffled order.

    The shuffle is deterministic and will always produce the same result given
    the same seed. The seed can be either provided explicitly or set via
    `ds.seed(seed)`. Prefer the latter if you don't need to control the shuffle
    seed individually. It allows to pass a single seed to derive seeds for all
    downstream random transformations in the pipeline.

    In multi-epoch training each epoch will be shuffled differently (i.e. the
    seed is combined with epoch number). In such case it is recommended to
    `shuffle` before `repeat` to avoid mixing elements from different epochs.

    Example usage:
    ```
    ds = MapDataset.range(5).shuffle()
    set(ds) == {0, 1, 2, 3, 4}
    list(ds) != [0, 1, 2, 3, 4]  # With probability (1 - 1/5!).
    ```

    Args:
      seed: An optional integer between 0 and 2**32-1 representing the seed used
        by the shuffling algorithm. If you don't need to control the shuffle
        seed individually, prefer setting the pipeline-level seed with
        `ds.seed(seed)` instead.

    Returns:
      A dataset containing the same elements but in a shuffled order.
    """
    # Loaded lazily due to a circular dependency (dataset <-> shuffle).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import shuffle
    # pylint: enable=g-import-not-at-top
    return shuffle.ShuffleMapDataset(parent=self, seed=seed)

  def slice(self, sl: builtins.slice) -> MapDataset[T]:
    """Returns a dataset containing only the elements with indices in `sl`.

    For most implementations of `MapDataset` slicing is also available through
    subscript operator: `list(ds.slice(slice(1, 10, 2))) == ds[1:10:2]`.

    Example usage:
    ```
    ds = MapDataset.range(5)
    list(ds.slice(slice(1, 3))) == [1, 2]
    list(ds.slice(slice(1, None, 2))) == [1, 3]
    ```

    Commonly used for sharding:
    `ds = ds.slice(slice(shard_index, None, shard_count))`, or, equivalently,
    `ds = ds[shard_index::shard_count]`.

    Args:
      sl: A `slice` object
        (https://docs.python.org/3/library/functions.html#slice) representing
        the slice of elements to that should constitute the returned dataset.

    Returns:
      A dataset containing only the elements with indices in the `sl` slice.
    """
    # Loaded lazily due to a circular dependency (dataset <-> slice).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        slice as slice_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return slice_dataset.SliceMapDataset(parent=self, sl=sl)

  def random_map(
      self,
      transform: (
          transforms.RandomMapTransform | Callable[[T, np.random.Generator], S]
      ),
      *,
      seed: int | None = None,
  ) -> MapDataset[S]:
    """Returns a dataset containing the elements transformed by `transform`.

    The `transform` is called with the element and a `np.random.Generator`
    instance that should be used inside the `transform` to preserve determinism.
    The seed can be either provided explicitly or set via `ds.seed(seed)`.
    Prefer the latter if you don't need to control the random map seed
    individually. It allows to pass a single seed to derive seeds for all
    downstream random transformations in the pipeline. The geenrator is seeded
    by a combination of the seed and the index of the element in the dataset.

    Example usage:
    ```
    ds = MapDataset.range(5)
    ds = ds.random_map(lambda x, rng: x + rng.integers(5, 10))
    set(ds).issubset(set(range(5, 15)))
    ```

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
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapMapDataset(
        parent=self, transform=transform, seed=seed
    )

  def repeat(self, num_epochs: int | None = None) -> MapDataset[T]:
    """Returns a dataset repeating the elements of this dataset multiple times.

    Specifying `None` for `num_epochs` will repeat the dataset infinitely, and
    causes `len(ds)` to return `sys.maxsize`.

    Since `MapDataset`s allow accessing elements past `len(ds) - 1` anyway
    (and use the index modulo `len(ds)`), this transformation effectively only
    changes the length of the dataset.

    Can not be called on an infinite dataset.

    Example usage:
    ```
    list(MapDataset.range(5).repeat(2)) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    ds = MapDataset.range(5).repeat()
    len(ds) == sys.maxsize
    ds[11111] == 1
    ```

    Args:
      num_epochs: Either a positive integer representing the number of times
        this dataset should be repeated or `None` to repeat infinitely.

    Returns:
      A dataset repeating the elements of this dataset multiple times.
    """
    # Loaded lazily due to a circular dependency (dataset <-> repeat).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import repeat
    # pylint: enable=g-import-not-at-top
    return repeat.RepeatMapDataset(parent=self, num_epochs=num_epochs)

  def to_iter_dataset(
      self,
      read_options: grain_options.ReadOptions | None = None,
      *,
      allow_nones: bool = False,
  ) -> IterDataset[T]:
    """Converts this dataset to an `IterDataset`.

    Elements from this dataset may be processed in multiple threads.

    Note that some of the transformations are not available on `IterDataset`.
    These are roughly transformations operating on element index such as
    `shuffle`, `map_with_index`, `slice` and `repeat`.

    Args:
      read_options: Controls multithreading when reading the data and applying
        transformations in this dataset.
      allow_nones: Whether to allow `None` values in the dataset (e.g. produced
        by `filter`). If `False` (the default), `None` values will be filtered
        out.

    Returns:
      An `IterDataset` with the same non-`None` elements as this dataset.
    """
    # Loaded lazily due to a circular dependency (dataset <-> prefetch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import prefetch
    # pylint: enable=g-import-not-at-top
    return prefetch.PrefetchIterDataset(
        self,
        read_options=read_options or grain_options.ReadOptions(),
        allow_nones=allow_nones,
    )

  def __iter__(self) -> DatasetIterator[T]:
    return self.to_iter_dataset().__iter__()

  @functools.cached_property
  def _stats(self) -> dataset_stats.Stats:
    """Returns the Stats object for recording statistics about this dataset."""
    # There may be parent `MapDataset` nodes introduced by users that did not
    # call super init and thus don't have `_parents`.
    parents_stats = []
    if hasattr(self, "_parents"):
      for p in self._parents:
        parents_stats.append(p._stats)  # pylint: disable=protected-access
    result = dataset_stats.make_stats(
        dataset_stats.StatsConfig(
            name=str(self), transform_mutates_spec=self._MUTATES_ELEMENT_SPEC
        ),
        parents_stats,
    )
    # `cached_property` is not thread-safe. By the time we finish constructing
    # the stats object, another thread may have already done it. So we check and
    # return it if that's the case. Note that this operation together with
    # `cached_property`s setter is not atomic so there's still a very small
    # chance of a race condition. However, consequences of the race are that the
    # optional stats collection will be disabled for the current transformation.
    # This is much less severe than the alternative of introducing lock
    # contention on the first batch processing.
    return self.__dict__.get("_stats", result)


class _IterDatasetMeta(abc.ABCMeta):
  """Metaclass for `IterDataset` containing factory transformations."""

  def mix(
      cls,
      datasets: Sequence[IterDataset[T]],
      weights: Sequence[float] | None = None,
  ) -> IterDataset[T]:
    """Returns a dataset that mixes input datasets with the given weights.

    NOTE: Stops producing elements once *any* input dataset is exhausted. If
    you need an infinite mixed dateset consider repeating the input datasets
    before mixing.

    Example usage:
    ```
    ds1 = MapDataset.range(5).to_iter_dataset()
    ds2 = MapDataset.range(7, 10).to_iter_dataset()
    list(IterDataset.mix([ds1, ds2])) == [0, 7, 1, 8, 2, 9, 3]
    ```

    Args:
      datasets: The datasets to mix.
      weights: The weights to use for mixing. Defaults to uniform weights if not
        specified.

    Returns:
      A dataset that represents a mixture of the input datasets according to the
      given weights.
    """
    # Loaded lazily due to a circular dependency (dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedIterDataset(parents=datasets, proportions=weights)


class IterDataset(_Dataset, Iterable[T], metaclass=_IterDatasetMeta):
  """Represents a dataset with transformations that support Iterable interface.

  Transformations do not mutate the dataset object. Instead, they return a new
  dataset. From the perspective of public APIs, `IterDataset` is immutable.
  """

  def __init__(
      self,
      parents: (
          MapDataset | IterDataset | Sequence[MapDataset | IterDataset]
      ) = (),
  ):
    if isinstance(parents, (MapDataset, IterDataset)):
      parents = (parents,)
    else:
      parents = tuple(parents)
    super().__init__(parents)
    self._parents = cast(
        Sequence[Union[MapDataset, IterDataset]], self._parents
    )
    usage_logging.log_event("IterDataset", tag_3="PyGrain")
    _api_usage_counter.Increment("IterDataset")

  @property
  def parents(self) -> Sequence[MapDataset | IterDataset]:
    return self._parents

  @property
  def _parent(self) -> MapDataset | IterDataset:
    assert len(self._parents) == 1, self._parents
    return self._parents[0]

  def batch(
      self,
      batch_size: int,
      *,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> IterDataset[S]:
    """Returns a dataset of elements batched along a new first dimension.

    Dataset elements are expected to be PyTrees.

    Example usage:
    ```
    ds = MapDataset.range(5).to_iter_dataset()
    ds = ds.batch(batch_size=2)
    list(ds) == [np.ndarray([0, 1]), np.ndarray([2, 3]), np.ndarray([4])]
    ```

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
    # Loaded lazily due to a circular dependency (dataset <-> batch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import batch
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
    return _WithSeedIterDataset(parent=self, seed=seed)

  def filter(
      self, transform: transforms.FilterTransform | Callable[[T], bool]
  ) -> IterDataset[T]:
    """Returns a dataset containing only the elements that match the filter.

    Example usage:
    ```
    ds = MapDataset.range(5).to_iter_dataset()
    ds = ds.filter(lambda x: x % 2 == 0)
    list(ds) == [0, 2, 4]
    ```

    Produces a warning if the filter removes more than 90% of the elements.
    You can adjust the threshold through
    `grain.experimental.DatasetOptions.filter_warn_threshold_ratio` used in
    `WithOptionsIterDataset`. In order to produce an exception in such case use
    `filter_raise_threshold_ratio`.

    Args:
      transform: Either a `FilterTransform` containing the `filter` method or a
        callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns `True`.
    """
    # Loaded lazily due to a circular dependency (dataset <-> filter).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        filter as filter_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return filter_dataset.FilterIterDataset(parent=self, transform=transform)

  def map(
      self, transform: transforms.MapTransform | Callable[[T], S]
  ) -> IterDataset[S]:
    """Returns a dataset containing the elements transformed by `transform`.

    Example usage:
    ```
    ds = MapDataset.range(5).to_iter_dataset()
    ds = ds.map(lambda x: x + 10)
    list(ds) == [10, 11, 12, 13, 14]
    ```

    Args:
      transform: Either a `MapTransform` containing the `map` method or a
        callable that takes an element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      `transform`.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapIterDataset(parent=self, transform=transform)

  def random_map(
      self,
      transform: (
          transforms.RandomMapTransform | Callable[[T, np.random.Generator], S]
      ),
      *,
      seed: int | None = None,
  ) -> IterDataset[S]:
    """Returns a dataset containing the elements transformed by `transform`.

    The `transform` is called with the element and a `np.random.Generator`
    instance that should be used inside the `transform` to preserve determinism.
    The seed can be either provided explicitly or set via `ds.seed(seed)`.
    Prefer the latter if you don't need to control the random map seed
    individually. It allows to pass a single seed to derive seeds for all
    downstream random transformations in the pipeline. The geenrator is seeded
    by a combination of the seed and a counter of elements produced by the
    dataset.

    Example usage:
    ```
    ds = MapDataset.range(5).to_iter_dataset()
    ds = ds.random_map(lambda x, rng: x + rng.integers(5, 10))
    set(ds).issubset(set(range(5, 15)))
    ```

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
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapIterDataset(
        parent=self, transform=transform, seed=seed
    )

  def prefetch(
      self, multiprocessing_options: grain_options.MultiprocessingOptions
  ) -> IterDataset[T]:
    """Deprecated, use `mp_prefetch` instead.

    Returns a dataset prefetching the elements in multiple processes.

    Each of the processes will process a slice of the dataset after all
    `MapDataset` transformations.

    WARNING: If the dataset contains many-to-one transformations (such as
    `batch`), output of prefetch may change if you change the number of workers.
    However, it is still going to be determisitic.

    Args:
      multiprocessing_options: options for the prefetching processes.
        `num_workers` must be greater than 0.

    Returns:
      A dataset prefetching input elements concurrently.
    """
    warnings.warn("Please use `mp_prefetch` instead.", DeprecationWarning)
    return self.mp_prefetch(multiprocessing_options)

  def mp_prefetch(
      self,
      options: grain_options.MultiprocessingOptions | None = None,
  ) -> IterDataset[T]:
    """Returns a dataset prefetching elements in multiple processes.

    Each of the processes works on a slice of the dataset. The slicing happens
    after all `MapDataset` transformations (right before `to_iter_dataset`).

    WARNING: If the dataset contains many-to-one transformations (such as
    `filter`) or stateful transformations (such as packing), output of
    `mp_prefetch` may change if `num_workers` is changed. However, it is still
    going to be determisitic. If you need elasticity in the number of prefetch
    workers, consider moving many-to-one and stateful transformations to after
    `mp_prefetch` or outside of the Grain pipeline.


    Args:
      options: options for the prefetching processes. `options.num_workers` must
        be greater than or equal to 0. If `options.num_workers` is 0,
        `mp_prefetch` has no effect. Defaults to
        `MultiprocessingOptions(num_workers=10)`.

    Returns:
      A dataset prefetching input elements in separate processes.
    """
    options = options or grain_options.MultiprocessingOptions(num_workers=10)
    # Loaded lazily due to a circular dependency (dataset <-> prefetch).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import prefetch
    # pylint: enable=g-import-not-at-top
    return prefetch.MultiprocessPrefetchIterDataset(
        self,
        multiprocessing_options=options,
    )

  @abc.abstractmethod
  def __iter__(self) -> DatasetIterator[T]:
    """Returns an iterator for this dataset."""


class DatasetIterator(Iterator[T], abc.ABC):
  """`IterDataset` iterator.

  NOTE: The methods are assumed to be thread-unsafe. Please ensure only a single
  thread can access a `DatasetIterator` instance.
  """

  # Whether this transformation mutates parent elements. This does not affect
  # the transformation itself, only used for information purposes in statistics.
  _MUTATES_ELEMENT_SPEC = True

  def __init__(
      self,
      parents: DatasetIterator | Sequence[DatasetIterator] = (),
  ):
    if isinstance(parents, DatasetIterator):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    # The options are set in `WithOptionsIterDataset`.
    self._options: DatasetOptions | None = None
    parent_options = []
    for p in self._parents:
      # Not all user iterators call super().__init__ and thus don't have the
      # options set.
      if (p_options := getattr(p, "_options", None)) is not None:
        parent_options.append(p_options)
    if parent_options:
      self._options = functools.reduce(lambda x, y: x.merge(y), parent_options)

  @property
  def _options_with_default(self) -> DatasetOptions:
    """Returns options for the pipeline including the given iterator.

    WARNING: must be accessed after all iterators in the pipeline have been
    initialized.
    """
    # TODO: Relax the requirement to access options after all iterators
    # in the pipeline have been initialized.
    return self._options or DatasetOptions()

  @property
  def _parent(self) -> DatasetIterator:
    assert len(self._parents) == 1, self._parents
    return self._parents[0]

  def __iter__(self) -> DatasetIterator[T]:
    return self

  # __next__ abstract method since we inherit from Iterator[T].

  @abc.abstractmethod
  def get_state(self) -> dict[str, Any]:
    """Returns the current state of the iterator.

    We reserve the right to evolve the state format over time. The states
    returned from this method are only guaranteed to be restorable by the same
    version of the code that produced them.

    Implementation Note: It is recommended that iterator implementations always
    produce states with the same shapes and types throughout the lifetime of the
    iterator. Some frameworks rely on this property to perform checkpointing,
    and all standard library iterators are compliant. It is also recommended to
    produce state values that support shapes and types, e.g. using `np.int64`
    instead of `int`. The standard library iterators are not currently compliant
    with this recommendation.
    """

  @abc.abstractmethod
  def set_state(self, state: dict[str, Any]):
    """Sets the current state of the iterator."""

  def start_prefetch(self) -> None:
    """Asynchronously starts processing and buffering elements.

    NOTE: Only available on iterators of asynchronous transformations.

    Can be useful when the iterator can be created in advance but the elements
    are not needed immediately. For instance, when recovering iterator and model
    from a checkpoint, recover the iterator first, call `start_prefech` and then
    recover the model. This way the time to get the first batch from the
    iterator will be partially or fully hidden behind the time it takes to
    recover the model.
    """
    raise NotImplementedError

  @functools.cached_property
  def _stats(self):
    """Returns the Stats object for recording statistics about this iterator."""
    # There may be parent `DatasetIterator` nodes introduced by users that did
    # not call super init and thus don't have `_stats`.
    parents_stats = []
    if hasattr(self, "_parents"):
      for p in self._parents:
        parents_stats.append(p._stats)  # pylint: disable=protected-access
    return dataset_stats.make_stats(
        dataset_stats.StatsConfig(
            name=str(self), transform_mutates_spec=self._MUTATES_ELEMENT_SPEC
        ),
        parents_stats,
    )


class _WithSeedMapDataset(MapDataset[T]):
  """Holds seed used by downstream transformations."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(self, parent: MapDataset[T], *, seed: int):
    super().__init__(parent)
    self._seed_rng_seed = seed

  def __len__(self) -> int:
    return self._parent.__len__()

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    return self._parent[index]

  def __str__(self):
    return "WithOptionsMapDataset"


class _WithSeedIterDataset(IterDataset[T]):
  """Holds seed used by downstream transformations."""

  def __init__(self, parent: IterDataset[T], *, seed: int):
    super().__init__(parent)
    self._seed_rng_seed = seed

  def __iter__(self) -> DatasetIterator[T]:
    return self._parent.__iter__()

  def __str__(self):
    return "WithOptionsIterDataset"


@dataclasses.dataclass(slots=True, frozen=True)
class _Default(Generic[T]):
  """Default options value holder."""

  value: T


@dataclasses.dataclass(kw_only=True, frozen=True)
class DatasetOptions:
  """Holds options used by dataset transformations."""

  # If the ratio of filtered out elements is above these thresholds, a warning
  # or an exception will be issued, respectively. Value `None` disables the
  # check.
  # The ratio is calculated on non-overlapping windows of 1000 elements. For
  # instancce, with `filter_warn_threshold_ratio=0.9` and 901 elements out of
  # the first 1000 (or elements 1000...2000) filtered out, a warning will be
  # issued.
  filter_warn_threshold_ratio: float | None | _Default[float] = _Default(0.9)
  filter_raise_threshold_ratio: float | None | _Default[None] = _Default(None)

  # Internal fields.

  # Names of fields which were set by the user.
  _user_set_fields: set[str] = dataclasses.field(
      default_factory=set, init=False
  )

  def __post_init__(self):
    # Replace default value objects with actual values.
    for field in dataclasses.fields(DatasetOptions):
      value = getattr(self, field.name)
      if isinstance(value, _Default):
        super().__setattr__(field.name, value.value)
      elif field.init:
        self._user_set_fields.add(field.name)

  def merge(self, other: DatasetOptions | None) -> DatasetOptions:
    """Merges these options with the other.

    Explicitly set options in `self` take precedence over options in `other`.

    Args:
      other: Options to merge.

    Returns:
      Merged options.
    """
    if other is None:
      return self

    merged = {}
    for field in dataclasses.fields(DatasetOptions):
      if field.name in self._user_set_fields:
        merged[field.name] = getattr(self, field.name)
      elif field.name in other._user_set_fields:  # pylint: disable=protected-access
        merged[field.name] = getattr(other, field.name)
    return DatasetOptions(**merged)


class WithOptionsIterDataset(IterDataset[T]):
  """Applies options to transformations in the pipeline.

  The options will apply to all transformations in the pipeline (before and
  after `WithOptionsIterDataset`). The options can be set multiple times in the
  pipeline, in which case they are merged. If the same option is set multiple
  times, the latest value takes precedence.

  Example:
  ```
  ds = MapDataset.range(5).to_iter_dataset()
  ds = WithOptionsIterDataset(
         ds,
         DatasetOptions(
             filter_warn_threshold_ratio=0.6,
             filter_raise_threshold_ratio=0.8,
         ),
       )
  ds = ds.filter(...)
  ds = WithOptionsIterDataset(
         ds,
         DatasetOptions(filter_warn_threshold_ratio=0.7),
       )
  ds = ds.filter(...)
  ```
  In this case, the options will be:
  ```
  filter_warn_threshold_ratio=0.7
  filter_raise_threshold_ratio=0.8
  ```
  """

  def __init__(self, parent: IterDataset[T], options: DatasetOptions):
    super().__init__(parent)
    self._options = options

  def __iter__(self) -> DatasetIterator[T]:
    result = self._parent.__iter__()
    # The parent iterator options are merged from the entire subtree. Merge
    # them with the latest options and update the subtree options.
    options = self._options.merge(result._options)
    to_visit = [result]
    while to_visit:
      current = to_visit.pop()
      current._options = options
      to_visit.extend(current._parents)
    return result

  def __str__(self):
    return f"WithOptionsIterDataset(options={self._options})"


_ConsistentDatasetType = TypeVar(
    "_ConsistentDatasetType", MapDataset, IterDataset
)


def apply_transformations(
    ds: _ConsistentDatasetType,
    transformations: transforms.Transformation | transforms.Transformations,
) -> _ConsistentDatasetType:
  """Applies transformations to a dataset.

  Args:
    ds: `MapDataset` or `IterDataset` to apply the transformations to.
    transformations: one or more transfromations to apply.

  Returns:
    Dataset of the same type with transformations applied.
  """
  if not isinstance(transformations, Sequence):
    transformations = (transformations,)
  for transformation in transformations:
    match transformation:
      case transforms.BatchTransform():
        ds = ds.batch(
            transformation.batch_size,
            drop_remainder=transformation.drop_remainder,
        )
      case transforms.MapTransform():
        ds = ds.map(transformation)
      case transforms.RandomMapTransform():
        ds = ds.random_map(transformation)
      case transforms.FilterTransform():
        ds = ds.filter(transformation)
      case _:
        raise NotImplementedError(
            f"Transformation type: {transformation} is not supported."
        )
  return ds
