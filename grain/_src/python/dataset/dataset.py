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
"""Dataset classes.

* ``MapDataset`` defines a dataset that supports efficient random access. It
  has 3 important methods:

  * ``__len__()`` returns the length of a single epoch over the dataset.

  * ``__getitem__()`` returns an element at the given positive index. The
    "true" length of a ``MapDataset`` is infinite.

  * Individual dataset elements are only evaluated when calling
    ``__getitem__()``. ``MapDataset`` s are stateless and will not hold
    elements.

* ``IterDataset`` defines a dataset that does not support efficient random
  access but can be iterated over. A ``MapDataset`` can be turned into
  a ``IterDataset`` but going from ``IterDataset`` to ``MapDataset`` is
  as expensive as materializing the whole dataset.

* ``DatasetIterator`` defines a stateful iterator of ``IterDataset``. The
  state of the iterator can be saved and restored.

Using the interfaces defined in ``collections.abc`` you can think of
``MapDataset`` as (infinite) ``Sequence``, ``IterDataset`` as ``Iterable`` and
``DatasetIterator`` as ``Iterator``.

``MapDataset`` is typically created by one of the factory methods in
``MapDatasetMeta`` (e.g. ``MapDataset.range(5)``). ``IterDataset`` is either
created by calling ``to_iter_dataset()`` on a ``MapDataset`` or by one of the
factory methods in ``IterDatasetMeta`` (e.g. ``IterDataset.mix([...])``).
"""

from __future__ import annotations

import abc
import builtins
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
import functools
import json
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings
import weakref

from etils import epath
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import checkpointing
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import stats as dataset_stats
from grain.proto import execution_summary_pb2
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
  """Node of a dataset tree structure that represents data transformation.

  Supports generating default seed for random transformations.
  """

  # Whether this transformation mutates parent elements. This does not affect
  # the transformation itself, only used for information purposes in statistics.
  _MUTATES_ELEMENT_SPEC = True

  def __init__(self, parents: Sequence[_Dataset]):
    super().__init__()
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

  # TODO: Define a more precise type signature for this method,
  # once pytype fully supports Concatenate and ParamSpec
  # (b/217789659, https://github.com/google/pytype/issues/786):
  # P = ParamSpec("P")
  # def pipe(
  #     self,
  #     func: Callable[Concatenate[Self, P], T],
  #     /,
  #     *args: P.args,
  #     **kwargs: P.kwargs,
  # ) -> T:
  def pipe(self, func: Callable[..., T], /, *args, **kwargs) -> T:
    """Syntactic sugar for applying a callable to this dataset.

    The ``pipe`` method, borrowed from ``pandas.DataFrame``, is convenient
    because it allows for using method chaining syntax in an extensible fashion,
    with transformations that are not built-in methods on ``Dataset``.

    For example, suppose you want to shuffle a dataset within a window.
    Functionality for this is available in ``WindowShuffleMapDataset``, but not
    as a method on ``MapDataset``, e.g.::

      dataset = (
          grain.experimental.WindowShuffleMapDataset(
              grain.MapDataset.range(1000),
              window_size=128,
              seed=0,
          )
          .batch(16)
      )

    This solution suffers from readability, because the shuffle transformation
    appears out of order from the data flow.

    In contrast, with ``pipe`` you can write::

      dataset = (
          grain.MapDataset.range(1000)
          .pipe(
              grain.experimental.WindowShuffleMapDataset,
              window_size=128,
              seed=0
          )
          .batch(16)
      )

    Args:
      func: The callable to apply to this dataset.
      *args: Additional positional arguments to pass to the callable.
      **kwargs: Keyword arguments to pass to the callable.

    Returns:
      The result of calling ``func(self, *args, **kwargs)``.
    """
    return func(self, *args, **kwargs)


class MapDatasetMeta(abc.ABCMeta):
  """Metaclass for ``MapDataset`` containing factory transformations."""

  def source(
      cls, source: Sequence[T] | base.RandomAccessDataSource[T]
  ) -> MapDataset[T]:
    """Returns a dataset that wraps a data source supporting random access.

    Example::

      ds = MapDataset.source(ArrayRecordDataSource(paths))

    Works with Sequence inputs as well::

      list(MapDataset.source([1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]

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

    Input arguments are interpreted the same way as in Python built-in
    ``range``:
      - ``range(n)`` => start=0, stop=n, step=1
      - ``range(m, n)`` => start=m, stop=n, step=1
      - ``range(m, n, p)`` => start=m, stop=n, step=p

    The produced values are consistent with the built-in `range` function::

      list(MapDataset.range(...)) == list(range(...))

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

    Example usage::

      ds1 = MapDataset.range(5)
      ds2 = MapDataset.range(7, 10)
      list(MapDataset.mix([ds1, ds2])) == [0, 7, 1, 8, 2, 9]

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

    Allows more general types of dataset mixing than ``mix``.

    Args:
      datasets: The datasets to select from.
      selection_map: Mapping from index within the mixed dataset to a selected
        dataset index and index within that dataset. Length of the resulting
        dataset will be determined by the length of the ``selection_map``.

    Returns:
      A MapDataset that represents a mixture of the input datasets according
      to the given selection map.
    """
    # Loaded lazily due to a circular dependency (dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.MixedMapDataset(parents=datasets, selection_map=selection_map)

  def concatenate(cls, datasets: Sequence[MapDataset[T]]) -> MapDataset[T]:
    """Returns a dataset of elements from all input datasets.

    Example usage::

      ds1 = MapDataset.range(3)
      ds2 = MapDataset.range(3, 8)
      list(MapDataset.concatenate([ds1, ds2])) == [0, 1, 2, 3, 4, 5, 6, 7]

    Args:
      datasets: The datasets to concatenate.

    Returns:
      A ``MapDataset`` that represents a concatenation of the input datasets.
      The n-th epoch of the returned dataset will be the n-th epoch of the
      component datasets.
    """
    # Loaded lazily due to a circular dependency (dataset <-> mix).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import mix
    # pylint: enable=g-import-not-at-top
    return mix.ConcatenateMapDataset(parents=datasets)


class MapDataset(_Dataset, Generic[T], metaclass=MapDatasetMeta):
  """Represents a dataset with transformations that support random access.

  Transformations do not mutate the dataset object. Instead, they return a new
  dataset. ``MapDataset`` is immutable.

  NOTE:
    ``MapDataset`` transformations such as ``.filter()`` use ``None`` to
    indicate absence of an element. Generally, the implementation of
    ``MapDataset`` transformations already handle `None` as a special case
    (e.g. by returning ``None`` as soon as ``__getitem__`` sees ``None``). This
    means the user-defined functions passed to the ``MapDataset``
    transformations do not need to explicitly handle ``None`` s.
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

    Example usage::

      ds = MapDataset.range(5)
      ds = ds.batch(batch_size=2)
      list(ds) == [np.ndarray([0, 1]), np.ndarray([2, 3]), np.ndarray([4])]

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
      self, transform: transforms.Filter | Callable[[T], bool]
  ) -> MapDataset[T]:
    """Returns a dataset containing only the elements that match the filter.

    Accessing an element of the returned dataset using subscription (``ds[i]``)
    returns:

    * ``None`` if ``transform`` returned ``False``
    * the element if ``transform`` returned ``True``

    Iterating over a filtered dataset skips ``None`` elements by default.

    Example usage::

      ds = MapDataset.range(5)
      ds = ds.filter(lambda x: x % 2 == 0)
      ds[2] == 2
      ds[1] == None
      list(ds) == [0, 2, 4]

    NOTE: ``list(ds)`` converts the dataset to an ``IterDataset`` with
    ``to_iter_dataset()`` under the hood which by default skips ``None``
    elements.

    ``to_iter_dataset`` produces a warning when iterating through a filtered
    dataset if the filter removes more than 90% of the elements.
    You can adjust the threshold through
    ``grain.experimental.DatasetOptions.filter_warn_threshold_ratio`` used in
    ``WithOptionsIterDataset``. In order to produce an exception in such case
    use ``filter_raise_threshold_ratio``.

    Args:
      transform: Either a ``FilterTransform`` containing the ``filter`` method
        or a callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns ``True``.
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
    """Returns a dataset containing the elements transformed by ``transform``.

    Example usage::

      ds = MapDataset.range(5)
      ds = ds.map(lambda x: x + 10)
      list(ds) == [10, 11, 12, 13, 14]

    Args:
      transform: Either a ``MapTransform`` containing the ``map`` method or a
        callable that takes an element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
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
      transform: transforms.MapWithIndex | Callable[[int, T], S],
  ) -> MapDataset[S]:
    """Returns a dataset containing the elements transformed by ``transform``.

    The transform is called with the index of the element within the dataset
    and the element itself.

    Example usage::

      ds = MapDataset.source(["a", "b", "c", "d"])
      ds = ds.map_with_index(lambda i, x: x + str(i))
      list(ds) == ["a0", "b1", "c2", "d3"]

    Args:
      transform: Either a ``MapWithIndexTransform`` containing the
        ``map_with_index`` method or a callable that takes an index and an
        element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
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

    When default seed generation is enabled by calling ``ds.seed``, every
    downstream random transformation will be automatically seeded with a unique
    seed by default. This simplifies seed management, making it easier to avoid:
     - Having to provide a seed in multiple transformations.
     - Accidentally reusing the same seed across transformations.

    It is recommended to call this right after the source. ``ds.seed`` has to be
    called before any random transformations (such as ``shuffle`` or
    ``random_map`` that rely on default seed generation to control their
    seeding). Given the same seed, the pipeline is guaranteed to always use the
    same seeds for each transformation.

    WARNING: The seed for random downstream transformations is derived from the
    seed passed to ``ds.seed`` and the absolute position of the transformation
    in the pipeline. This means that if you add transformations before the
    random transformation, its seed will change. For instance, if this random
    transformation is ``shuffle``, adding a transformation before ``shuffle``
    will change its seed and data order, consequently. To avoid this, pass the
    seed to the transformation directly.

    Note about custom dataset implementations: the default seed generation is
    available through ``_default_seed``, but the private API is not guaranteed
    to be stable.

    Example 1::

      ds = ds.seed(seed).shuffle()

    ``shuffle`` will automatically derive its own seed (different from
    ``seed``).

    Example 2::

      ds = ds.seed(seed).shuffle().random_map(...)

    ``shuffle`` and ``random_map`` will each derive their own seed and the seeds
    are going to be different.

    Example 3::

      ds = ds.seed(seed).random_map(transform, seed=seed1)

    ``random_map`` will use ``seed1`` and will not be affected by ``seed``. This
    can be used to control individual transformation seeding independently from
    the rest of the pipeline.

    Example 4::

      ds = ds.seed(seed1).shuffle().seed(seed2).random_map(...)

    ``ds.seed`` only affects the downstream transformations and can be
    overridden by a subsequent ``seed`` call.
    ``shuffle`` will derive its seed from ``seed1``, ``random_map`` - from
    ``seed2`` and will not be affected by ``seed1``. This can be used to control
    your transformation seeding even if you don't own the first part of the
    pipeline.

    Example 5::

      ds1 = ds.source(...).seed(seed1).shuffle()
      ds2 = ds.source(...).seed(seed2).shuffle()
      ds = MapDataset.mix([ds1, ds2], ...).random_map(...)

    Each ``shuffle`` will derive its own seed from ``seed1`` or ``seed2``
    respectively. ``random_map`` will derive its seed from both ``seed1`` and
    ``seed2``.

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
    ``ds.seed(seed)``. Prefer the latter if you don't need to control the
    shuffle seed individually. It allows to pass a single seed to derive seeds
    for all downstream random transformations in the pipeline.

    In multi-epoch training each epoch will be shuffled differently (i.e. the
    seed is combined with epoch number). In such case it is recommended to
    ``shuffle`` before ``repeat`` to avoid mixing elements from different
    epochs.

    Example usage::

      ds = MapDataset.range(5).shuffle()
      set(ds) == {0, 1, 2, 3, 4}
      list(ds) != [0, 1, 2, 3, 4]  # With probability (1 - 1/5!).


    Args:
      seed: An optional integer between 0 and 2**32-1 representing the seed used
        by the shuffling algorithm. If you don't need to control the shuffle
        seed individually, prefer setting the pipeline-level seed with
        ``ds.seed(seed)`` instead.

    Returns:
      A dataset containing the same elements but in a shuffled order.
    """
    # Loaded lazily due to a circular dependency (dataset <-> shuffle).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import shuffle
    # pylint: enable=g-import-not-at-top
    return shuffle.ShuffleMapDataset(parent=self, seed=seed)

  def slice(self, sl: builtins.slice) -> MapDataset[T]:
    """Returns a dataset containing only the elements with indices in ``sl``.

    For most implementations of ``MapDataset`` slicing is also available through
    subscript operator: ``list(ds.slice(slice(1, 10, 2))) == ds[1:10:2]``.

    Example usage::

      ds = MapDataset.range(5)
      list(ds.slice(slice(1, 3))) == [1, 2]
      list(ds.slice(slice(1, None, 2))) == [1, 3]


    Commonly used for sharding:
    ``ds = ds.slice(slice(shard_index, None, shard_count))``, or, equivalently,
    ``ds = ds[shard_index::shard_count]``.

    Args:
      sl: A ``slice`` object
        (https://docs.python.org/3/library/functions.html#slice) representing
        the slice of elements to that should constitute the returned dataset.

    Returns:
      A dataset containing only the elements with indices in the ``sl`` slice.
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
    """Returns a dataset containing the elements transformed by ``transform``.

    The ``transform`` is called with the element and a ``np.random.Generator``
    instance that should be used inside the ``transform`` to preserve
    determinism. The seed can be either provided explicitly or set via
    ``ds.seed(seed)``. Prefer the latter if you don't need to control the random
    map seed individually. It allows to pass a single seed to derive seeds for
    all downstream random transformations in the pipeline. The generator is
    seeded by a combination of the seed and the index of the element in the
    dataset.

    NOTE: Avoid using the provided RNG outside of the ``transform`` function
    (e.g. by passing it to the next transformation along with the data).
    The RNG is going to be reused.

    Example usage::

      ds = MapDataset.range(5)
      ds = ds.random_map(lambda x, rng: x + rng.integers(5, 10))
      set(ds).issubset(set(range(5, 15)))


    Args:
      transform: Either a ``RandomMapTransform`` containing the ``random_map``
        method or a callable that takes an element and a np.random.Generator and
        returns a new element.
      seed: An optional integer between 0 and 2**32-1 representing the seed used
        to initialize the random number generator used by ``transform``. If you
        don't need to control the shuffle seed individually, prefer setting the
        pipeline-level seed with``ds.seed(seed)`` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.RandomMapMapDataset(
        parent=self, transform=transform, seed=seed
    )

  def repeat(self, num_epochs: int | None = None) -> MapDataset[T]:
    """Returns a dataset repeating the elements of this dataset multiple times.

    Specifying ``None`` for ``num_epochs`` will repeat the dataset infinitely,
    and causes ``len(ds)`` to return ``sys.maxsize``.

    Since ``MapDataset`` allows accessing elements past ``len(ds) - 1`` anyway
    (and uses the index modulo ``len(ds)``), this transformation effectively
    only changes the length of the dataset.

    Can not be called on an infinite dataset.

    Example usage::

      list(MapDataset.range(5).repeat(2)) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
      ds = MapDataset.range(5).repeat()
      len(ds) == sys.maxsize
      ds[11111] == 1


    Args:
      num_epochs: Either a positive integer representing the number of times
        this dataset should be repeated or ``None`` to repeat infinitely.

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
    """Converts this dataset to an ``IterDataset``.

    Elements from this dataset may be processed in multiple threads.

    Note that some of the transformations are not available on ``IterDataset``.
    These are roughly transformations operating on element index such as
    ``shuffle``, ``map_with_index``, ``slice`` and ``repeat``.

    Args:
      read_options: Controls multithreading when reading the data and applying
        transformations in this dataset.
      allow_nones: Whether to allow ``None`` values in the dataset (e.g.
        produced by ``filter``). If ``False`` (the default), ``None`` values
        will be filtered out.

    Returns:
      An ``IterDataset`` with the same non- ``None`` elements as this dataset.
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

  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ) -> dataset_stats.Stats:
    """Eagerly initializes the stats object with given execution tracking mode.

    Sets the `_stats` attribute with specified execution tracking mode,
    bypassing the `_stats` cached property.
    This is beneficial when we want to initialize the stats object eagerly from
    PrefetchDatasetIterator, using the appropriate execution tracking mode from
    the grain options.

    Args:
      execution_tracking_mode: The execution tracking mode to use for the stats
        object.

    Returns:
      The initialized stats object.
    """
    # There may be parent `MapDataset` nodes introduced by users that did not
    # call super init and thus don't have `_parents`.
    parents_stats = []
    if hasattr(self, "_parents"):
      for p in self._parents:
        parents_stats.append(p._initialize_stats(execution_tracking_mode))  # pylint: disable=protected-access
    self._stats = dataset_stats.make_stats(
        dataset_stats.StatsConfig(
            name=str(self),
            transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
            iter_weakref=weakref.ref(self),
        ),
        parents_stats,
        execution_tracking_mode=execution_tracking_mode,
    )
    return self._stats

  @functools.cached_property
  def _stats(self) -> dataset_stats.Stats:
    """Returns the Stats object for recording statistics about this dataset."""
    return self._initialize_stats(base.ExecutionTrackingMode.DISABLED)


class IterDatasetMeta(abc.ABCMeta):
  """Metaclass for ``IterDataset`` containing factory transformations."""

  def mix(
      cls,
      datasets: Sequence[IterDataset[T]],
      weights: Sequence[float] | None = None,
  ) -> IterDataset[T]:
    """Returns a dataset that mixes input datasets with the given weights.

    NOTE: Stops producing elements once *any* input dataset is exhausted. If
    you need an infinite mixed dateset consider repeating the input datasets
    before mixing.

    Example usage::

      ds1 = MapDataset.range(5).to_iter_dataset()
      ds2 = MapDataset.range(7, 10).to_iter_dataset()
      list(IterDataset.mix([ds1, ds2])) == [0, 7, 1, 8, 2, 9, 3]


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


class IterDataset(_Dataset, Iterable[T], metaclass=IterDatasetMeta):
  """Represents a dataset with transformations that support Iterable interface.

  Transformations do not mutate the dataset object. Instead, they return a new
  dataset. ``IterDataset`` is immutable.
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

    Example usage::

      ds = MapDataset.range(5).to_iter_dataset()
      ds = ds.batch(batch_size=2)
      list(ds) == [np.ndarray([0, 1]), np.ndarray([2, 3]), np.ndarray([4])]


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

    When default seed generation is enabled by calling ``ds.seed``, every
    downstream random transformation will be automatically seeded with a unique
    seed by default. This simplifies seed management, making it easier to avoid:
     - Having to provide a seed in multiple transformations.
     - Accidentally reusing the same seed across transformations.

    It is recommended to call this right after the source. ``ds.seed`` has to be
    called before any random transformations (such as ``random_map`` that rely
    on default seed generation to control their seeding). Given the same seed,
    the pipeline is guaranteed to always use the same seeds for each
    transformation.

    Note about custom dataset implementations: the default seed generation is
    available through ``_default_seed``, but the private API is not guaranteed
    to be stable.

    Example 1::

      ds = ds.seed(seed).random_map(...)

    ``random_map`` will automatically derive its own seed (different from
    ``seed``).

    Example 2::

      ds = ds.seed(seed).random_map().random_map(...)

    The first and second ``random_map`` s will each derive their own seed and
    the seeds are going to be different.

    Example 3::

      ds = ds.seed(seed).random_map(transform, seed=seed1)

    ``random_map`` will use ``seed1`` and will not be affected by ``seed``. This
    can be used to control individual transformation seeding independently from
    the rest of the pipeline.

    Example 4::

      ds = ds.seed(seed1).random_map(...).seed(seed2).random_map(...)

    ``ds.seed`` only affects the downstream transformations and can be
    overridden by a subsequent ``seed`` call.
    The first ``random_map`` will derive its seed from ``seed1``, the second
    - from ``seed2`` and will not be affected by ``seed1``. This can be used to
    control your transformation seeding even if you don't own the first part of
    the pipeline.

    Example 5::

      ds1 = ds.source(...).seed(seed2).shuffle().to_iter_dataset()
      ds2 = ds.source(...).seed(seed2).shuffle().to_iter_dataset()
      ds = IterDataset.mix([ds1, ds2], ...).random_map(...)

    Each ``shuffle`` will derive its own seed from ``seed1`` or ``seed2``
    respectively. ``random_map`` will derive its seed from both ``seed1`` and
    ``seed2``.

    Args:
      seed: Seed to use.

    Returns:
      A dataset with elements unchanged.
    """
    return _WithSeedIterDataset(parent=self, seed=seed)

  def filter(
      self, transform: transforms.Filter | Callable[[T], bool]
  ) -> IterDataset[T]:
    """Returns a dataset containing only the elements that match the filter.

    Example usage::

      ds = MapDataset.range(5).to_iter_dataset()
      ds = ds.filter(lambda x: x % 2 == 0)
      list(ds) == [0, 2, 4]


    Produces a warning if the filter removes more than 90% of the elements.
    You can adjust the threshold through
    ``grain.experimental.DatasetOptions.filter_warn_threshold_ratio`` used in
    ``WithOptionsIterDataset``. In order to produce an exception in such case
    use ``filter_raise_threshold_ratio``.

    Args:
      transform: Either a ``FilterTransform`` containing the ``filter`` method
        or a callable that takes an element and returns a boolean.

    Returns:
      A dataset of the same type containing only the elements for which the
      filter transform returns ``True`` .
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
    """Returns a dataset containing the elements transformed by ``transform``.

    Example usage::

      ds = MapDataset.range(5).to_iter_dataset()
      ds = ds.map(lambda x: x + 10)
      list(ds) == [10, 11, 12, 13, 14]

    Args:
      transform: Either a ``MapTransform`` containing the ``map`` method or a
        callable that takes an element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
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
    """Returns a dataset containing the elements transformed by ``transform``.

    The ``transform`` is called with the element and a ``np.random.Generator``
    instance that should be used inside the ``transform`` to preserve
    determinism.
    The seed can be either provided explicitly or set via ``ds.seed(seed)``.
    Prefer the latter if you don't need to control the random map seed
    individually. It allows to pass a single seed to derive seeds for all
    downstream random transformations in the pipeline. The geenrator is seeded
    by a combination of the seed and a counter of elements produced by the
    dataset.

    NOTE: Avoid using the provided RNG outside of the ``transform`` function
    (e.g. by passing it to the next transformation along with the data).
    The RNG is going to be reused.

    Example usage::

      ds = MapDataset.range(5).to_iter_dataset()
      ds = ds.random_map(lambda x, rng: x + rng.integers(5, 10))
      set(ds).issubset(set(range(5, 15)))


    Args:
      transform: Either a ``RandomMapTransform`` containing the ``random_map``
        method or a callable that takes an element and a np.random.Generator and
        returns a new element.
      seed: An integer between 0 and 2**32-1 representing the seed used to
        initialize the random number generator used by ``transform``. If you
        don't need to control the transformation seed individually, prefer
        setting the pipeline-level seed with ``ds.seed(seed)`` instead.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.RandomMapIterDataset(
        parent=self, transform=transform, seed=seed
    )

  def map_with_index(
      self, transform: transforms.MapWithIndex | Callable[[int, T], S]
  ) -> IterDataset[S]:
    """Returns a dataset of the elements transformed by the ``transform``.

    The ``transform`` is called with the index of the element in the dataset
    as the first argument and the element as the second argument.

    Example usage::

      ds = MapDataset.range(5).to_iter_dataset()
      ds = ds.map(lambda i, x: (i, 2**x))
      list(ds) == [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16)]

    Args:
      transform: Either a ``MapWithIndexTransform`` containing the
        ``map_with_index`` method or a callable that takes an index and an
        element and returns a new element.

    Returns:
      A dataset containing the elements of the original dataset transformed by
      ``transform``.
    """
    # Loaded lazily due to a circular dependency (dataset <-> map).
    # pylint: disable=g-import-not-at-top
    from grain._src.python.dataset.transformations import (
        map as map_dataset,
    )
    # pylint: enable=g-import-not-at-top
    return map_dataset.MapWithIndexIterDataset(parent=self, transform=transform)

  def prefetch(
      self, multiprocessing_options: grain_options.MultiprocessingOptions
  ) -> IterDataset[T]:
    """Deprecated, use ``mp_prefetch`` instead.

    Returns a dataset prefetching the elements in multiple processes.

    Each of the processes will process a slice of the dataset after all
    ``MapDataset`` transformations.

    WARNING: If the dataset contains many-to-one transformations (such as
    ``batch``), output of prefetch may change if you change the number of
    workers. However, it is still going to be determisitic.

    Args:
      multiprocessing_options: options for the prefetching processes.
        ``num_workers`` must be greater than 0.

    Returns:
      A dataset prefetching input elements concurrently.
    """
    warnings.warn("Please use `mp_prefetch` instead.", DeprecationWarning)
    return self.mp_prefetch(multiprocessing_options)

  def mp_prefetch(
      self,
      options: grain_options.MultiprocessingOptions | None = None,
      worker_init_fn: Callable[[int, int], None] | None = None,
  ) -> IterDataset[T]:
    """Returns a dataset prefetching elements in multiple processes.

    Each of the processes works on a slice of the dataset. The slicing happens
    after all ``MapDataset`` transformations (right before ``to_iter_dataset``).

    WARNING: If the dataset contains many-to-one transformations (such as
    ``filter``) or stateful transformations (such as packing), output of
    ``mp_prefetch`` may change if ``num_workers`` is changed. However, it is
    still going to be determisitic. If you need elasticity in the number of
    prefetch workers, consider moving many-to-one and stateful transformations
    to after ``mp_prefetch`` or outside of the Grain pipeline.


    Args:
      options: options for the prefetching processes. ``options.num_workers``
        must be greater than or equal to 0. If ``options.num_workers`` is 0,
        ``mp_prefetch`` has no effect. Defaults to
        ``MultiprocessingOptions(num_workers=10)``.
      worker_init_fn: A function that is called in each worker process before
        the data is processed. The function takes two arguments: the current
        worker index and the total worker count.

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
        worker_init_fn=worker_init_fn,
    )

  @abc.abstractmethod
  def __iter__(self) -> DatasetIterator[T]:
    """Returns an iterator for this dataset."""


class DatasetIterator(Iterator[T], abc.ABC):
  """``IterDataset`` iterator.

  NOTE: The methods are assumed to be thread-unsafe. Please ensure only a single
  thread can access a ``DatasetIterator`` instance.
  """

  # Whether this transformation mutates parent elements. This does not affect
  # the transformation itself, only used for information purposes in statistics.
  _MUTATES_ELEMENT_SPEC = True

  def __init__(
      self,
      parents: DatasetIterator | Sequence[DatasetIterator] = (),
  ):
    super().__init__()
    if isinstance(parents, DatasetIterator):
      self._parents = (parents,)
    else:
      self._parents = tuple(parents)
    if self._parents:
      self._ctx: base.IteratorContext = self._parents[0]._ctx
      # Merge the context from all parents.
      to_visit = list(self._parents[1:])
      for parent in to_visit:
        self._ctx.merge(parent._ctx)
      # Update the context in the parent iterator trees.
      while to_visit:
        current = to_visit.pop()
        current._ctx = self._ctx
        to_visit.extend(current._parents)
    else:
      self._ctx: base.IteratorContext = base.IteratorContext()

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
    produce state values that support shapes and types, e.g. using ``np.int64``
    instead of ``int``. The standard library iterators are not currently
    compliant with this recommendation.
    """

  @abc.abstractmethod
  def set_state(self, state: dict[str, Any]):
    """Sets the current state of the iterator."""

  def start_prefetch(self) -> None:
    """Asynchronously starts processing and buffering elements.

    NOTE: Only available on iterators of asynchronous transformations.

    Can be useful when the iterator can be created in advance but the elements
    are not needed immediately. For instance, when recovering iterator and model
    from a checkpoint, recover the iterator first, call ``start_prefech`` and
    then recover the model. This way the time to get the first batch from the
    iterator will be partially or fully hidden behind the time it takes to
    recover the model.
    """
    raise NotImplementedError

  @functools.cached_property
  def _stats(self):
    """Returns the Stats object for recording statistics about this iterator."""
    config = dataset_stats.StatsConfig(
        name=str(self),
        transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
        iter_weakref=weakref.ref(self),
    )
    return dataset_stats.make_stats(
        config,
        [p._stats for p in self._parents],  # pylint: disable=protected-access
        execution_tracking_mode=(
            self._ctx.dataset_options.execution_tracking_mode
        ),
    )

  ### BEGIN Orbax checkpointing API.
  # See orbax.checkpoint.v1.handlers.StatefulCheckpointable for more details.
  # See https://orbax.readthedocs.io/en/latest/ for usage examples.

  async def save(
      self, directory: checkpointing.PathAwaitingCreation
  ) -> Awaitable[None]:
    """Saves the iterator state to a directory.

    The current state (`get_state`) is used for saving, so any updates to the
    state after returning from this method will not affect the saved checkpoint.

    Args:
      directory: A path in the process of being created. Must call
        await_creation before accessing the physical path.

    Returns:
      A coroutine that has not been awaited. This is called by Orbax in a
      background thread to perform I/O without blocking the main thread.
    """
    state = json.dumps(self.get_state(), indent=4)
    return checkpointing.background_save(directory, state)

  async def load(self, directory: epath.Path) -> Awaitable[None]:
    """Loads the iterator state from a directory.

    The state may be loaded and set in a background thread. The main thread
    should not alter the state content while the load is in progress.

    Args:
      directory: The directory to load the state from.

    Returns:
      A coroutine that has not been awaited. This is called by Orbax in a
      background thread to perform I/O without blocking the main thread.
    """

    def set_state_fn(state: str):
      self.set_state(json.loads(state))

    return checkpointing.background_load(directory, set_state_fn)

  ### END Orbax checkpointing API.


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


class WithOptionsIterDataset(IterDataset[T]):
  """Applies options to transformations in the pipeline.

  The options will apply to all transformations in the pipeline (before and
  after `WithOptionsIterDataset`). The options can be set multiple times in the
  pipeline, in which case they are merged. If the same option is set multiple
  times, the latest value takes precedence.

  Example::

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

  In this case, the options will be::

    filter_warn_threshold_ratio=0.7
    filter_raise_threshold_ratio=0.8
  """

  def __init__(self, parent: IterDataset[T], options: base.DatasetOptions):
    super().__init__(parent)
    self.options = options

  def __iter__(self) -> DatasetIterator[T]:
    result = self._parent.__iter__()
    # The parent iterator options are merged from the entire subtree. Merge
    # them with the latest options and update the subtree options.
    options = self.options.merge(result._ctx.dataset_options)
    result._ctx.dataset_options = options
    return result

  def __str__(self):
    return f"WithOptionsIterDataset(options={self.options})"


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
    transformations: one or more transformations to apply.

  Returns:
    Dataset of the same type with transformations applied.
  """
  if not isinstance(transformations, Sequence):
    transformations = (transformations,)
  for transformation in transformations:
    match transformation:
      case transforms.Batch():
        ds = ds.batch(
            transformation.batch_size,
            drop_remainder=transformation.drop_remainder,
        )
      case transforms.MapTransform():
        ds = ds.map(transformation)
      case transforms.RandomMapTransform():
        ds = ds.random_map(transformation)
      case transforms.MapWithIndex():
        ds = ds.map_with_index(transformation)
      case transforms.FlatMapTransform():
        # Loaded lazily due to a circular dependency (dataset <-> flatmap).
        # pylint: disable=g-import-not-at-top
        from grain._src.python.dataset.transformations import flatmap
        # pylint: enable=g-import-not-at-top
        if isinstance(ds, MapDataset):
          ds = flatmap.FlatMapMapDataset(ds, transformation)
        else:
          ds = flatmap.FlatMapIterDataset(ds, transformation)
      case transforms.Filter():
        ds = ds.filter(transformation)
      case _:
        raise NotImplementedError(
            f"Transformation type: {transformation} is not supported."
        )
  return ds


def get_execution_summary(
    ds: DatasetIterator,
) -> execution_summary_pb2.ExecutionSummary:
  """Returns the execution summary for the dataset."""
  # pylint: disable=protected-access
  execution_stats = ds._stats
  if not isinstance(execution_stats, dataset_stats._ExecutionStats):
    raise ValueError(
        "Set `grain_py_debug_mode` or set `execution_tracking_mode` in grain"
        " options to `STAGE_TIMING` to enable execution statistics collection."
    )
  return execution_stats._get_execution_summary()
  # pylint: enable=protected-access
