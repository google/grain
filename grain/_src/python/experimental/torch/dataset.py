# Copyright 2026 Google LLC
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
"""PyTorch-first `MapDataset`, `IterDataset`, and `DatasetIterator` facades."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Never, NoReturn, TypeVar, overload
import warnings

from grain._src.core import transforms
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.experimental.torch import batch as torch_batch
from grain._src.python.experimental.torch import checkpoint as torch_checkpoint
from grain._src.python.experimental.torch import prefetch as torch_prefetch
from grain._src.python.experimental.torch import random_map as torch_random_map

T = TypeVar("T")
S = TypeVar("S")

_RANDOM_MAP_NOT_SUPPORTED_MESSAGE = (
    "`random_map` and `RandomMap` transformations are not supported by"
    " `grain.experimental.torch` datasets, because they pass a NumPy random"
    " number generator to the transformation. Use `stateless_random_map`, which"
    " passes a PyTorch PRNG key, instead."
)


# The PyTorch `MapDataset` and `IterDataset` below are thin facades: each one
# holds a plain `grain.MapDataset` or `grain.IterDataset` in `_inner` and builds
# every transformation on it. The pipeline is therefore made of plain Grain
# datasets only, with a single facade on the outside that supplies the PyTorch
# defaults. The helpers below maintain that invariant at the boundaries between
# the facades and plain Grain:
#
# - Unwrap before handing a dataset to plain Grain (the facade constructors and
#   factories such as `mix` and `concatenate`), so that a facade never ends up
#   nested inside the pipeline or inside another facade.
# - Wrap what comes back from user code that was given the wrapped dataset
#   (`pipe`), so that the PyTorch defaults keep applying to the rest of the
#   chain.


def _unwrap_map(ds: dataset.MapDataset[T]) -> dataset.MapDataset[T]:
  """Returns the plain Grain dataset behind ``ds``, or ``ds`` if it is one."""
  if isinstance(ds, MapDataset):
    return ds._inner  # pylint: disable=protected-access
  return ds


def _unwrap_iter(ds: dataset.IterDataset[T]) -> dataset.IterDataset[T]:
  """Returns the plain Grain dataset behind ``ds``, or ``ds`` if it is one."""
  if isinstance(ds, IterDataset):
    return ds._inner  # pylint: disable=protected-access
  return ds


def _wrap_dataset_if_needed(obj: Any) -> Any:
  """Returns ``obj`` as its PyTorch facade if it is a plain Grain dataset."""
  if isinstance(obj, (MapDataset, IterDataset)):
    return obj
  if isinstance(obj, dataset.MapDataset):
    return MapDataset(obj)
  if isinstance(obj, dataset.IterDataset):
    return IterDataset(obj)
  return obj


def _pipe(
    ds: MapDataset[Any] | IterDataset[Any],
    func: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
  """Returns `func(ds, *args, **kwargs)`, as a PyTorch dataset if a dataset."""
  first_arg: Any = ds
  if isinstance(func, type) and issubclass(
      func, (dataset.MapDataset, dataset.IterDataset)
  ):
    # A dataset class is built on the wrapped dataset, as the methods of `ds`
    # build their transformations, so that the pipeline behaves exactly as in
    # plain Grain. Any other callable gets `ds` itself, so that the methods it
    # calls on `ds` have the PyTorch defaults.
    first_arg = ds._inner  # pylint: disable=protected-access
  return _wrap_dataset_if_needed(func(first_arg, *args, **kwargs))


def _raise_unsupported_transformation(transformation: Never) -> NoReturn:
  """Raises ``NotImplementedError`` for a transformation that cannot be applied.

  The ``Never`` parameter makes type checkers report a member of
  ``transforms.Transformation`` that ``_apply_torch_transformations`` does not
  handle, like ``typing.assert_never``, while the error raised at runtime
  remains ``NotImplementedError``.

  Args:
    transformation: The transformation that is not supported.
  """
  raise NotImplementedError(
      f"Transformation type: {transformation} is not supported."
  )


def _apply_torch_transformations(
    ds: MapDataset[Any] | IterDataset[Any],
    transformations: transforms.Transformation | transforms.Transformations,
) -> Any:
  """Applies transformations while preserving PyTorch defaults."""
  if not isinstance(transformations, Sequence):
    transformations = (transformations,)
  for transformation in transformations:
    match transformation:
      case transforms.Batch():
        ds = ds.batch(
            transformation.batch_size,
            drop_remainder=transformation.drop_remainder,
            batch_fn=transformation.batch_fn,
        )
      case transforms.Map():
        ds = ds.map(transformation)
      case transforms.RandomMap():
        raise NotImplementedError(_RANDOM_MAP_NOT_SUPPORTED_MESSAGE)
      case transforms.StatelessRandomMap():
        ds = ds.stateless_random_map(transformation)
      case transforms.MapWithIndex():
        ds = ds.map_with_index(transformation)
      case transforms.FlatMap():
        # pylint: disable=g-import-not-at-top
        from grain._src.python.dataset.transformations import flatmap
        # pylint: enable=g-import-not-at-top
        if isinstance(ds, MapDataset):
          ds = MapDataset(
              flatmap.FlatMapMapDataset(ds._inner, transformation)  # pylint: disable=protected-access
          )
        elif isinstance(ds, IterDataset):
          ds = IterDataset(
              flatmap.FlatMapIterDataset(ds._inner, transformation)  # pylint: disable=protected-access
          )
        else:
          raise TypeError(
              "Expected a `grain.experimental.torch.MapDataset` or"
              f" `IterDataset`, got {type(ds).__name__}."
          )
      case transforms.Filter():
        ds = ds.filter(transformation)
      case transforms.TfRandomMap():
        # Not supported by `apply`, as in plain Grain.
        raise NotImplementedError(
            f"Transformation type: {transformation} is not supported."
        )
      case _:
        _raise_unsupported_transformation(transformation)
  return ds


class DatasetIterator(dataset.DatasetIterator[T]):
  """``grain.DatasetIterator`` that supports PyTorch checkpointing.

  Supports the same methods as ``grain.DatasetIterator``; see its documentation
  for the full reference. Only the differences are documented here:

  - ``state_dict`` and ``load_state_dict`` make the iterator a
    ``torch.distributed.checkpoint`` ``Stateful`` object, so it can be passed
    to ``dcp.save`` and ``dcp.load`` next to the model and the optimizer. The
    state is stored per rank, so every rank restores its own position.

  Example usage::

    import torch.distributed.checkpoint as dcp

    it = iter(ds)  # `ds` is a `grain.experimental.torch.IterDataset`.
    ...
    dcp.save({"model": model, "data": it}, checkpoint_id=path)

    # To resume, rebuild `ds` and restore the position of a new iterator.
    it = iter(ds)
    dcp.load({"model": model, "data": it}, checkpoint_id=path)

  ``state_dict`` and ``load_state_dict`` use the default process group. If the
  ranks that read different data form another group, such as the data-parallel
  group under tensor parallelism, wrap the iterator in
  ``IteratorCheckpointWrapper`` to pass that group::

    data = grain_torch.IteratorCheckpointWrapper(it, group=dp_group)
    dcp.save({"model": model, "data": data}, checkpoint_id=path)

  Like any ``grain.DatasetIterator``, this iterator closes the iterator that it
  wraps when it is closed or garbage-collected. To checkpoint an iterator of a
  ``grain.MapDataset`` or ``grain.IterDataset``, wrap it in
  ``IteratorCheckpointWrapper`` rather than in this class.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(self, parent: dataset.DatasetIterator[T]):
    if not isinstance(parent, dataset.DatasetIterator):
      raise TypeError(
          "`DatasetIterator` expects a `grain.DatasetIterator` instance, got"
          f" {type(parent).__name__}."
      )
    super().__init__(parent)

  def __next__(self) -> T:
    self._assert_not_closed()
    return next(self._parent)

  def get_state(self) -> dict[str, Any]:
    return self._parent.get_state()

  def set_state(self, state: dict[str, Any]) -> None:
    self._parent.set_state(state)

  def state_dict(self) -> dict[str, Any]:
    """Returns a rank-keyed checkpoint dictionary for ``dcp.save``."""
    return torch_checkpoint.IteratorCheckpointWrapper(self).state_dict()

  def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
    """Restores iterator state from a ``dcp.load`` dictionary."""
    torch_checkpoint.IteratorCheckpointWrapper(self).load_state_dict(state_dict)

  def _get_next_index(self) -> int:
    return dataset.get_next_index(self._parent)

  def _set_next_index(self, index: int) -> None:
    dataset.set_next_index(self._parent, index)

  # This iterator only forwards to its parent, so it shares the parent's
  # statistics instead of being a stage of its own. Initializing statistics of
  # its own would re-create those of all ancestors and discard what they have
  # recorded.
  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ) -> dataset_stats.Stats:
    return self._parent._initialize_stats(execution_tracking_mode)  # pylint: disable=protected-access

  @property
  def _stats(self) -> dataset_stats.Stats:
    return self._parent._stats  # pylint: disable=protected-access

  def __str__(self) -> str:
    return f"TorchDatasetIterator({self._parent})"


# pylint: disable=no-self-argument
class _TorchMapDatasetMeta(dataset.MapDatasetMeta):
  """Metaclass providing factory methods returning PyTorch ``MapDataset``s."""

  def source(
      cls, source: Sequence[T] | base.RandomAccessDataSource[T]
  ) -> MapDataset[T]:
    return cls(dataset.MapDataset.source(source))  # pyrefly: ignore[bad-return]

  def range(
      cls, start: int, stop: int | None = None, step: int = 1
  ) -> MapDataset[int]:
    return cls(dataset.MapDataset.range(start, stop, step))  # pyrefly: ignore[bad-return]

  def mix(
      cls,
      datasets: Sequence[dataset.MapDataset[T]],
      weights: Sequence[float] | None = None,
  ) -> MapDataset[T]:
    unwrapped = [_unwrap_map(d) for d in datasets]
    return cls(dataset.MapDataset.mix(unwrapped, weights=weights))  # pyrefly: ignore[bad-return]

  def select_from_datasets(
      cls,
      datasets: Sequence[dataset.MapDataset[T]],
      selection_map: base.DatasetSelectionMap,
  ) -> MapDataset[T]:
    unwrapped = [_unwrap_map(d) for d in datasets]
    return cls(
        dataset.MapDataset.select_from_datasets(
            unwrapped, selection_map=selection_map
        )
    )  # pyrefly: ignore[bad-return]

  def concatenate(
      cls, datasets: Sequence[dataset.MapDataset[T]]
  ) -> MapDataset[T]:
    unwrapped = [_unwrap_map(d) for d in datasets]
    return cls(dataset.MapDataset.concatenate(unwrapped))  # pyrefly: ignore[bad-return]


# pylint: enable=no-self-argument


class MapDataset(dataset.MapDataset[T], metaclass=_TorchMapDatasetMeta):
  """``grain.MapDataset`` with PyTorch defaults.

  Supports the same methods as ``grain.MapDataset``, including the factory
  methods such as ``source`` and ``range``; see its documentation for the full
  reference and examples. Only the differences are documented here:

  - ``stateless_random_map`` passes PyTorch PRNG keys to plain callables
    (``TorchStatelessRandomMap``).
  - ``random_map`` is not supported, because it passes a NumPy generator to the
    transformation. Use ``stateless_random_map`` instead.
  - ``batch`` collates with ``torch.utils.data.default_collate`` by default.
  - Transformations return ``grain.experimental.torch.MapDataset``, and
    ``to_iter_dataset`` returns ``grain.experimental.torch.IterDataset``.

  Each instance wraps a plain ``grain.MapDataset`` and builds every
  transformation on it, so default seeds, slicing, and shared-memory batching
  behave exactly as in ``grain.MapDataset``.

  Example usage::

    import torch
    from grain.experimental import torch as grain_torch

    ds = (
        grain_torch.MapDataset.source([torch.zeros(3)] * 8)
        .shuffle(seed=42)
        .batch(4)
    )
    ds[0].shape  # torch.Size([4, 3])
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(self, inner: dataset.MapDataset[T]):
    if not isinstance(inner, dataset.MapDataset):
      raise TypeError(
          "`MapDataset` expects a `grain.MapDataset` instance, got"
          f" {type(inner).__name__}."
      )
    unwrapped = _unwrap_map(inner)
    super().__init__(unwrapped)
    self._inner: dataset.MapDataset[T] = unwrapped

  @classmethod
  def from_dataset(cls, ds: dataset.MapDataset[T]) -> MapDataset[T]:
    """Returns ``ds`` as a ``grain.experimental.torch.MapDataset``.

    Use it to get the PyTorch defaults for a pipeline that was built with
    ``grain.MapDataset``, for example by a library function. ``ds`` is returned
    unchanged if it already is a ``grain.experimental.torch.MapDataset``.

    Example usage::

      ds = grain.MapDataset.range(4)  # A plain Grain dataset.
      ds = grain_torch.MapDataset.from_dataset(ds)
      ds = ds.batch(2)  # Batches are `torch.Tensor`s.

    Args:
      ds: A ``grain.MapDataset``.

    Returns:
      A ``grain.experimental.torch.MapDataset`` over the same pipeline.
    """
    if isinstance(ds, cls):
      return ds
    return cls(ds)

  def __len__(self) -> int:
    return len(self._inner)

  @overload
  def __getitem__(self, index: builtins.slice) -> MapDataset[T]:
    ...

  @overload
  def __getitem__(self, index: int) -> T | None:
    ...

  def __getitem__(self, index):
    if isinstance(index, builtins.slice):
      return self.slice(index)
    return self._inner[index]

  def _getitems(self, indices: Sequence[int]) -> Sequence[T | None]:
    return self._inner._getitems(indices)  # pylint: disable=protected-access

  def seed(self, seed: int) -> MapDataset[T]:
    return MapDataset(self._inner.seed(seed))

  def shuffle(self, seed: int | None = None) -> MapDataset[T]:
    return MapDataset(self._inner.shuffle(seed=seed))

  def slice(
      self,
      sl: builtins.slice | Sequence[int],
      wrap_around: bool = False,
  ) -> MapDataset[T]:
    if isinstance(sl, MapDataset):
      sl = _unwrap_map(sl)  # pyrefly: ignore[bad-assignment]
    return MapDataset(self._inner.slice(sl, wrap_around=wrap_around))

  def repeat(
      self, num_epochs: int | None = None, *, reseed_each_epoch: bool = True
  ) -> MapDataset[T]:
    return MapDataset(
        self._inner.repeat(
            num_epochs=num_epochs, reseed_each_epoch=reseed_each_epoch
        )
    )

  def filter(
      self, transform: transforms.Filter | Callable[[T], bool]
  ) -> MapDataset[T]:
    return MapDataset(self._inner.filter(transform))

  def map(self, transform: transforms.Map | Callable[[T], S]) -> MapDataset[S]:
    return MapDataset(self._inner.map(transform))

  def map_with_index(
      self, transform: transforms.MapWithIndex | Callable[[int, T], S]
  ) -> MapDataset[S]:
    return MapDataset(self._inner.map_with_index(transform))

  def random_map(
      self,
      transform: transforms.RandomMap | Callable[..., Any],
      *,
      seed: int | None = None,
  ) -> NoReturn:
    """Not supported; use ``stateless_random_map`` instead.

    ``grain.MapDataset.random_map`` passes a NumPy generator to ``transform``,
    which makes it easy to mix NumPy arrays into a PyTorch pipeline.
    ``stateless_random_map`` passes a PyTorch PRNG key instead.

    Args:
      transform: Unused.
      seed: Unused.

    Raises:
      NotImplementedError: Always.
    """
    del transform, seed  # Unused.
    raise NotImplementedError(_RANDOM_MAP_NOT_SUPPORTED_MESSAGE)

  def stateless_random_map(
      self,
      transform: transforms.StatelessRandomMap | Callable[[T, Any], S],
      *,
      seed: int | None = None,
  ) -> MapDataset[Any]:
    """Returns a dataset transformed with a per-element PyTorch PRNG key.

    Same as ``grain.MapDataset.stateless_random_map``, except that a plain
    callable ``fn(element, key)`` is wrapped in ``TorchStatelessRandomMap``
    instead of ``JaxStatelessRandomMap``: ``key`` is a key tensor for the
    stateless random functions in ``torch.func._random``. A
    ``StatelessRandomMap`` instance is used as is.

    Example usage::

      import torch
      import torch.func._random as torch_random
      from grain.experimental import torch as grain_torch

      def add_noise(x, key):
        return x + torch_random.normal(key, x.shape)

      ds = grain_torch.MapDataset.source([torch.zeros(2)] * 4)
      ds = ds.seed(42).stateless_random_map(add_noise)
      # The key depends only on the seed and the element's index, so the
      # output is the same on every run.

    Args:
      transform: A callable ``fn(element, key)`` (wrapped in
        ``TorchStatelessRandomMap``) or a ``StatelessRandomMap`` implementation.
      seed: An optional non-negative 64-bit integer seed. Defaults to the seed
        derived from ``ds.seed(...)``.

    Returns:
      A ``MapDataset`` containing the transformed elements.
    """
    stateless_transform = torch_random_map.as_torch_stateless_random_map(
        transform, "stateless_random_map"
    )
    return MapDataset(
        self._inner.stateless_random_map(stateless_transform, seed=seed)
    )

  def batch(
      self,
      batch_size: int,
      *,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> MapDataset[Any]:
    """Returns a dataset of elements batched along a new first dimension.

    Same as ``grain.MapDataset.batch``, except that ``batch_fn`` defaults to
    ``torch.utils.data.default_collate``. Batches are ``torch.Tensor``s; NumPy
    arrays and Python numbers are converted to tensors too.

    Example usage::

      ds = grain_torch.MapDataset.range(5)
      ds = ds.batch(batch_size=2)
      list(ds)  # [tensor([0, 1]), tensor([2, 3]), tensor([4])]

    Args:
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to ``torch.utils.data.default_collate``.

    Returns:
      A dataset of batches. A batch has the PyTree structure of an element,
      except that ``torch.utils.data.default_collate`` returns a list for a
      tuple.
    """
    effective_batch_fn: Any = (
        torch_batch.default_collate if batch_fn is None else batch_fn
    )
    return MapDataset(
        self._inner.batch(
            batch_size,
            drop_remainder=drop_remainder,
            batch_fn=effective_batch_fn,
        )
    )

  def apply(
      self,
      transformations: transforms.Transformation | transforms.Transformations,
  ) -> MapDataset[Any]:
    return _apply_torch_transformations(self, transformations)

  def pipe(self, func: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Returns ``func(self, *args, **kwargs)``.

    Same as ``grain.MapDataset.pipe``, except that a ``grain.MapDataset`` or
    ``grain.IterDataset`` returned by ``func`` is returned as a
    ``grain.experimental.torch`` dataset. This way, plain Grain dataset classes
    can be used in a PyTorch pipeline. Such a class is applied to the
    ``grain.MapDataset`` that this dataset wraps.

    Example usage::

      ds = grain_torch.MapDataset.range(8)
      ds = ds.pipe(
          grain.experimental.WindowShuffleMapDataset, window_size=4, seed=0
      )
      ds = ds.batch(2)  # Batches are `torch.Tensor`s.

    Args:
      func: A callable that takes this dataset as its first argument.
      *args: Positional arguments to pass to ``func`` after the dataset.
      **kwargs: Keyword arguments to pass to ``func``.

    Returns:
      The result of ``func``. A ``grain.MapDataset`` or ``grain.IterDataset`` is
      returned as a ``grain.experimental.torch`` dataset.
    """
    return _pipe(self, func, *args, **kwargs)

  def to_iter_dataset(
      self,
      read_options: grain_options.ReadOptions | None = None,
      *,
      allow_nones: bool = False,
  ) -> IterDataset[T]:
    return IterDataset(
        self._inner.to_iter_dataset(
            read_options=read_options, allow_nones=allow_nones
        )
    )

  def __iter__(self) -> DatasetIterator[T]:
    return self.to_iter_dataset().__iter__()

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._inner)

  # This dataset only forwards to `_inner`, so it shares the statistics of
  # `_inner` instead of being a stage of its own.
  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ) -> dataset_stats.Stats:
    return self._inner._initialize_stats(execution_tracking_mode)  # pylint: disable=protected-access

  @property
  def _stats(self) -> dataset_stats.Stats:
    return self._inner._stats  # pylint: disable=protected-access

  def __str__(self) -> str:
    return f"TorchMapDataset({self._inner})"


# pylint: disable=no-self-argument
class _TorchIterDatasetMeta(dataset.IterDatasetMeta):
  """Metaclass providing factory methods returning PyTorch ``IterDataset``s."""

  def mix(
      cls,
      datasets: (
          Sequence[dataset.IterDataset[T]]
          | Mapping[str, dataset.IterDataset[T]]
      ),
      weights: Sequence[float] | Mapping[str, float] | None = None,
  ) -> IterDataset[T]:
    if isinstance(datasets, Mapping):
      unwrapped: Any = {k: _unwrap_iter(v) for k, v in datasets.items()}
    else:
      unwrapped = [_unwrap_iter(d) for d in datasets]
    return cls(dataset.IterDataset.mix(unwrapped, weights=weights))  # pyrefly: ignore[bad-return]


# pylint: enable=no-self-argument


class IterDataset(dataset.IterDataset[T], metaclass=_TorchIterDatasetMeta):
  """``grain.IterDataset`` with PyTorch defaults.

  Supports the same methods as ``grain.IterDataset``; see its documentation for
  the full reference and examples. Only the differences are documented here:

  - ``stateless_random_map`` passes PyTorch PRNG keys to plain callables
    (``TorchStatelessRandomMap``).
  - ``random_map`` is not supported, because it passes a NumPy generator to the
    transformation. Use ``stateless_random_map`` instead.
  - ``batch`` collates with ``torch.utils.data.default_collate`` by default.
  - ``mp_prefetch`` limits PyTorch threads in worker processes and passes CPU
    tensors between processes through Grain's shared memory.
  - ``with_options`` applies ``DatasetOptions``.
  - Transformations return ``grain.experimental.torch.IterDataset``, and
    iterating returns ``grain.experimental.torch.DatasetIterator``.

  Each instance wraps a plain ``grain.IterDataset`` and builds every
  transformation on it, so the pipeline behaves exactly as in
  ``grain.IterDataset``.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(self, inner: dataset.IterDataset[T]):
    if not isinstance(inner, dataset.IterDataset):
      raise TypeError(
          "`IterDataset` expects a `grain.IterDataset` instance, got"
          f" {type(inner).__name__}."
      )
    unwrapped = _unwrap_iter(inner)
    # Pass empty parents to `super().__init__()` so `unwrapped.has_child` stays
    # `False`, which lets `_OutputIterDataset` run inside `unwrapped.__iter__`.
    super().__init__(())
    self._inner: dataset.IterDataset[T] = unwrapped
    self._parents = (unwrapped,)

  # `has_child` is always `True`, so that Grain does not treat this dataset as
  # the end of the pipeline and `DatasetIterator` remains outermost.
  @property
  def has_child(self) -> bool:
    return True

  @has_child.setter
  def has_child(self, value: bool) -> None:
    # Grain sets `has_child` when a dataset is built on top of this one.
    # Then the wrapped dataset is not the end of the pipeline either.
    if value:
      self._inner.has_child = True

  @classmethod
  def from_dataset(cls, ds: dataset.IterDataset[T]) -> IterDataset[T]:
    """Returns ``ds`` as a ``grain.experimental.torch.IterDataset``.

    Use it to get the PyTorch defaults for a pipeline that was built with
    ``grain.IterDataset``, for example by a library function. ``ds`` is returned
    unchanged if it already is a ``grain.experimental.torch.IterDataset``.

    Example usage::

      ds = grain.MapDataset.range(4).to_iter_dataset()  # A plain Grain dataset.
      ds = grain_torch.IterDataset.from_dataset(ds)
      ds = ds.batch(2)  # Batches are `torch.Tensor`s.

    Args:
      ds: A ``grain.IterDataset``.

    Returns:
      A ``grain.experimental.torch.IterDataset`` over the same pipeline.
    """
    if isinstance(ds, cls):
      return ds
    return cls(ds)

  def set_slice(
      self, sl: builtins.slice, sequential_slice: bool = False
  ) -> None:
    """Restricts this dataset, in place, to a slice of its elements.

    Called by ``mp_prefetch`` on each worker's copy of the dataset to give the
    worker its share of the data. It is not meant to be called directly.

    Args:
      sl: The slice of elements to keep.
      sequential_slice: Whether the slice is a contiguous range of the source
        rather than every n-th element.
    """
    dataset.set_slice(self._inner, sl, sequential_slice)

  def enable_shared_memory_output(self) -> None:
    """Makes this dataset write its output to shared memory, if it can.

    Called by ``mp_prefetch`` on the dataset that it prefetches from. It is not
    meant to be called directly.
    """
    if isinstance(self._inner, base.SupportsSharedMemoryOutput):
      self._inner.enable_shared_memory_output()

  def seed(self, seed: int) -> IterDataset[T]:
    return IterDataset(self._inner.seed(seed))

  def filter(
      self, transform: transforms.Filter | Callable[[T], bool]
  ) -> IterDataset[T]:
    return IterDataset(self._inner.filter(transform))

  def map(self, transform: transforms.Map | Callable[[T], S]) -> IterDataset[S]:
    return IterDataset(self._inner.map(transform))

  def map_with_index(
      self, transform: transforms.MapWithIndex | Callable[[int, T], S]
  ) -> IterDataset[S]:
    return IterDataset(self._inner.map_with_index(transform))

  def random_map(
      self,
      transform: transforms.RandomMap | Callable[..., Any],
      *,
      seed: int | None = None,
  ) -> NoReturn:
    """Not supported; use ``stateless_random_map`` instead.

    ``grain.IterDataset.random_map`` passes a NumPy generator to ``transform``,
    which makes it easy to mix NumPy arrays into a PyTorch pipeline.
    ``stateless_random_map`` passes a PyTorch PRNG key instead.

    Args:
      transform: Unused.
      seed: Unused.

    Raises:
      NotImplementedError: Always.
    """
    del transform, seed  # Unused.
    raise NotImplementedError(_RANDOM_MAP_NOT_SUPPORTED_MESSAGE)

  def stateless_random_map(
      self,
      transform: transforms.StatelessRandomMap | Callable[[T, Any], S],
      *,
      seed: int | None = None,
  ) -> IterDataset[Any]:
    """Returns a dataset transformed with a per-element PyTorch PRNG key.

    Same as ``grain.IterDataset.stateless_random_map``, except that a plain
    callable ``fn(element, key)`` is wrapped in ``TorchStatelessRandomMap``
    instead of ``JaxStatelessRandomMap``: ``key`` is a key tensor for the
    stateless random functions in ``torch.func._random``. A
    ``StatelessRandomMap`` instance is used as is.

    Example usage::

      import torch
      import torch.func._random as torch_random
      from grain.experimental import torch as grain_torch

      def add_noise(x, key):
        return x + torch_random.normal(key, x.shape)

      ds = grain_torch.MapDataset.source([torch.zeros(2)] * 4).to_iter_dataset()
      ds = ds.seed(42).stateless_random_map(add_noise)
      # The key depends only on the seed and the element's index, so the
      # output is the same on every run.

    Args:
      transform: A callable ``fn(element, key)`` (wrapped in
        ``TorchStatelessRandomMap``) or a ``StatelessRandomMap`` implementation.
      seed: An optional non-negative 64-bit integer seed. Defaults to the seed
        derived from ``ds.seed(...)``.

    Returns:
      An ``IterDataset`` containing the transformed elements.
    """
    stateless_transform = torch_random_map.as_torch_stateless_random_map(
        transform, "stateless_random_map"
    )
    return IterDataset(
        self._inner.stateless_random_map(stateless_transform, seed=seed)
    )

  def batch(
      self,
      batch_size: int,
      *,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[T]], S] | None = None,
  ) -> IterDataset[Any]:
    """Returns a dataset of elements batched along a new first dimension.

    Same as ``grain.IterDataset.batch``, except that ``batch_fn`` defaults to
    ``torch.utils.data.default_collate``. Batches are ``torch.Tensor``s; NumPy
    arrays and Python numbers are converted to tensors too.

    Example usage::

      ds = grain_torch.MapDataset.range(5).to_iter_dataset()
      ds = ds.batch(batch_size=2)
      list(ds)  # [tensor([0, 1]), tensor([2, 3]), tensor([4])]

    Args:
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to ``torch.utils.data.default_collate``.

    Returns:
      A dataset of batches. A batch has the PyTree structure of an element,
      except that ``torch.utils.data.default_collate`` returns a list for a
      tuple.
    """
    effective_batch_fn: Any = (
        torch_batch.default_collate if batch_fn is None else batch_fn
    )
    return IterDataset(
        self._inner.batch(
            batch_size,
            drop_remainder=drop_remainder,
            batch_fn=effective_batch_fn,
        )
    )

  def with_options(self, options: base.DatasetOptions) -> IterDataset[T]:
    """Returns a dataset with ``options`` applied to the pipeline.

    Method form of ``grain.experimental.WithOptionsIterDataset`` that keeps the
    result a ``grain.experimental.torch.IterDataset``. The options apply to all
    transformations in the pipeline, before and after this call. Options set
    by a later call take precedence.

    Example usage::

      ds = ds.to_iter_dataset()
      # Tensors smaller than 64 KiB are copied between processes instead of
      # being placed in shared memory.
      ds = ds.with_options(grain_torch.DatasetOptions(min_shm_size=64 << 10))
      ds = ds.mp_prefetch(grain_torch.MultiprocessingOptions(num_workers=4))

    Args:
      options: The options to apply.

    Returns:
      An ``IterDataset`` with the options applied.
    """
    return IterDataset(dataset.WithOptionsIterDataset(self._inner, options))

  def prefetch(
      self, multiprocessing_options: grain_options.MultiprocessingOptions
  ) -> IterDataset[T]:
    warnings.warn("Please use `mp_prefetch` instead.", DeprecationWarning)
    return self.mp_prefetch(multiprocessing_options)

  def mp_prefetch(
      self,
      options: grain_options.MultiprocessingOptions | None = None,
      worker_init_fn: Callable[[int, int], None] | None = None,
      sequential_slice: bool = False,
  ) -> IterDataset[T]:
    """Returns a dataset prefetching elements in multiple processes.

    Same as ``grain.IterDataset.mp_prefetch``; see its documentation for how
    workers slice the dataset and for cleanup recommendations. There are two
    PyTorch-specific differences:

    - Each worker process is limited to ``options.worker_intra_op_threads``
      PyTorch intra-op threads (1 by default), like
      ``torch.utils.data.DataLoader`` workers. The limit is applied before
      ``worker_init_fn`` runs. To use another thread count, set
      ``worker_intra_op_threads`` rather than calling ``torch.set_num_threads``
      in ``worker_init_fn``, which some PyTorch builds ignore after the first
      call.
    - CPU tensors are passed from the workers through Grain's shared memory,
      so ``DatasetOptions.min_shm_size`` applies to them as it does to NumPy
      arrays.

    Example usage::

      ds = grain_torch.MapDataset.source([torch.zeros(1024)] * 8).batch(4)
      ds = ds.to_iter_dataset().mp_prefetch(
          grain_torch.MultiprocessingOptions(num_workers=2)
      )
      it = iter(ds)
      next(it).shape  # torch.Size([4, 1024])
      it.close()

    Args:
      options: Options for the prefetching processes, either
        ``grain.experimental.torch.MultiprocessingOptions`` or
        ``grain.MultiprocessingOptions``. If ``options.num_workers`` is 0,
        ``mp_prefetch`` has no effect. Defaults to 10 workers with one intra-op
        thread each.
      worker_init_fn: A function that is called in each worker process before
        the data is processed. The function takes two arguments: the current
        worker index and the total worker count.
      sequential_slice: Whether the slice for each worker should be sequential
        (consecutive at the source dataset). Experimental.

    Returns:
      A dataset prefetching input elements in separate processes.
    """
    return IterDataset(
        torch_prefetch.multiprocess_prefetch(
            self._inner,
            options=options,
            worker_init_fn=worker_init_fn,
            sequential_slice=sequential_slice,
        )
    )

  def apply(
      self,
      transformations: transforms.Transformation | transforms.Transformations,
  ) -> IterDataset[Any]:
    return _apply_torch_transformations(self, transformations)

  def pipe(self, func: Callable[..., Any], /, *args, **kwargs) -> Any:
    """Returns ``func(self, *args, **kwargs)``.

    Same as ``grain.IterDataset.pipe``, except that a ``grain.MapDataset`` or
    ``grain.IterDataset`` returned by ``func`` is returned as a
    ``grain.experimental.torch`` dataset. This way, plain Grain dataset classes
    can be used in a PyTorch pipeline. Such a class is applied to the
    ``grain.IterDataset`` that this dataset wraps.

    Example usage::

      ds = grain_torch.MapDataset.range(8).to_iter_dataset()
      ds = ds.pipe(
          grain.experimental.WindowShuffleIterDataset, window_size=4, seed=0
      )
      ds = ds.batch(2)  # Batches are `torch.Tensor`s.

    Args:
      func: A callable that takes this dataset as its first argument.
      *args: Positional arguments to pass to ``func`` after the dataset.
      **kwargs: Keyword arguments to pass to ``func``.

    Returns:
      The result of ``func``. A ``grain.MapDataset`` or ``grain.IterDataset`` is
      returned as a ``grain.experimental.torch`` dataset.
    """
    return _pipe(self, func, *args, **kwargs)

  def __iter__(self) -> DatasetIterator[T]:
    return DatasetIterator(self._inner.__iter__())

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._inner)

  def __str__(self) -> str:
    return f"TorchIterDataset({self._inner})"


@overload
def mp_prefetch(
    ds: IterDataset[T],
    options: grain_options.MultiprocessingOptions | None = None,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> IterDataset[T]:
  ...


@overload
def mp_prefetch(
    ds: dataset.IterDataset[T],
    options: grain_options.MultiprocessingOptions | None = None,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> dataset.IterDataset[T]:
  ...


def mp_prefetch(
    ds: dataset.IterDataset[T],
    options: grain_options.MultiprocessingOptions | None = None,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> dataset.IterDataset[T]:
  """Returns a dataset prefetching the elements of ``ds`` in multiple processes.

  The function form of ``grain.experimental.torch.IterDataset.mp_prefetch``;
  see its documentation. It can also be applied to a ``grain.IterDataset``, for
  example with ``grain.IterDataset.pipe``, and returns the same kind of dataset
  as ``ds``.

  Args:
    ds: The dataset to prefetch from: a ``grain.experimental.torch.IterDataset``
      or a ``grain.IterDataset``.
    options: Options for the prefetching processes, either
      ``grain.experimental.torch.MultiprocessingOptions`` or
      ``grain.MultiprocessingOptions``. Defaults to 10 workers with one intra-op
      thread each.
    worker_init_fn: A function that is called in each worker process before the
      data is processed. The function takes two arguments: the current worker
      index and the total worker count.
    sequential_slice: Whether the slice for each worker should be sequential
      (consecutive at the source dataset). Experimental.

  Returns:
    A dataset prefetching input elements in separate processes: a
    ``grain.experimental.torch.IterDataset`` if ``ds`` is one, and a
    ``grain.IterDataset`` otherwise.

  Raises:
    TypeError: If ``ds`` is not a ``grain.IterDataset``, for example if it is a
      ``MapDataset``.
  """
  if isinstance(ds, IterDataset):
    # Use the torch version of multiprocess prefetch if the dataset is
    # already a torch IterDataset.
    return ds.mp_prefetch(
        options=options,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    )
  elif isinstance(ds, dataset.IterDataset):
    # Wrap the grain.IterDataset in the torch version of multiprocess prefetch.
    return torch_prefetch.multiprocess_prefetch(
        ds,
        options=options,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    )
  else:
    raise TypeError(
        "`mp_prefetch` expects a `grain.IterDataset` instance, got"
        f" {type(ds).__name__}. Call `to_iter_dataset()` on a `MapDataset`"
        " first."
    )
