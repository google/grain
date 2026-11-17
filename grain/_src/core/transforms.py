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
"""Abstract base classes for common types of transformations.

The idea is to implement atomic chunks of transformations as subclasses of
the base classes below (examples: resize image, tokenize text, add padding).
- More complex transformations can be created by chaining multiple
  transformations.
- It's recommended that subclasses are dataclasses.
- Libraries applying transformations can use these base classes to correctly
  apply transformations as part of Beam pipelines, data ingestion pipelines etc.
"""
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
import dataclasses
import inspect
from typing import Any, Protocol, Union, runtime_checkable

import numpy as np


class Map(abc.ABC):
  """Abstract base class for all 1:1 transformations of elements.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Example:
    Applying a custom map transformation to increment each element::

      import grain

      # Define a custom map transformation to add one to each element.
      class AddOne(grain.transforms.Map):
        def map(self, element: int) -> int:
          return element + 1

      # Create a parent dataset.
      parent_ds = grain.MapDataset.range(5)
      print(list(parent_ds))
      # [0, 1, 2, 3, 4]

      # Apply the map transformation.
      transformed_ds = parent_ds.map(AddOne())
      print(list(transformed_ds))
      # [1, 2, 3, 4, 5]
  """

  @abc.abstractmethod
  def map(self, element):
    """Maps a single element.

    Args:
      element: The input element to transform.

    Returns:
      The transformed output element.
    """


class RandomMap(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Example:
    Applying a random map transformation to add noise::

      import grain
      import numpy as np

      # Define a custom random map transformation to add random offset.
      class AddRandomOffset(grain.transforms.RandomMap):
        def random_map(self, element: int, rng: np.random.Generator) -> int:
          return element + int(rng.integers(0, 10))

      # Create a parent dataset and seed it for deterministic results.
      parent_ds = grain.MapDataset.range(5).seed(42)
      print(list(parent_ds))
      # [0, 1, 2, 3, 4]

      # Apply the random map transformation.
      transformed_ds = parent_ds.random_map(AddRandomOffset())
      print(list(transformed_ds))
      # [2, 5, 5, 4, 9]
  """

  @abc.abstractmethod
  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element using a random number generator.

    Args:
      element: The input element to transform.
      rng: A NumPy random generator initialized deterministically for this
        element.

    Returns:
      The randomly transformed output element.
    """


class MapWithIndex(abc.ABC):
  """Abstract base class for 1:1 transformations of elements and their index.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Example:
    Applying a transformation that pairs each element with its index::

      import grain

      # Define a custom transform that inherits from MapWithIndex and implements
      # the map_with_index method.
      class AddIndex(grain.transforms.MapWithIndex):
        def map_with_index(self, index: int, element: int) -> tuple[int, int]:
          return index + element * 10

      # Create a parent dataset.
      parent_ds = grain.MapDataset.range(3)
      print(list(parent_ds))
      # [0, 1, 2]

      # Apply the MapWithIndex transformation to the parent dataset.
      transformed_ds = parent_ds.map_with_index(AddIndex())
      print(list(transformed_ds))
      # [0, 11, 22]
  """

  @abc.abstractmethod
  def map_with_index(self, index: int, element):
    """Maps a single element with its index.

    Args:
      index: The zero-based index of the element within the dataset.
      element: The input element to transform.

    Returns:
      The transformed output element.
    """


class TfRandomMap(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements."""

  @abc.abstractmethod
  def np_random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""


@runtime_checkable
class StatelessRandomMap(Protocol):
  """Protocol for stateless key-based random 1:1 transformations.

  Implementations must be pure and thread-safe because transformations execute
  concurrently across prefetch threads and worker processes. Different backends
  (such as JAX, PyTorch, TensorFlow, or custom PRNGs) implement this protocol
  without requiring framework imports inside Grain core.
  """

  def make_key_for_index(self, seed: int) -> Callable[[int], Any]:
    """Returns a pure function ``(index: int) -> key`` derived from ``seed``."""
    ...

  def stateless_random_map(self, element: Any, key: Any) -> Any:
    """Maps ``element`` deterministically using an immutable ``key``."""
    ...


def _split_uint64(value: int) -> tuple[np.uint32, np.uint32]:
  """Returns the (low, high) uint32 words of the low 64 bits of `value`."""
  return np.uint32(value & 0xFFFFFFFF), np.uint32((value >> 32) & 0xFFFFFFFF)


class JaxStatelessRandomMap:
  """Stateless JAX random map implementing ``StatelessRandomMap``.

  Can be used in two ways:
  1. Wrapping a callable ``fn(element, key)``: ``JaxStatelessRandomMap(fn)``
     (used by default when passing a plain callable to
     ``ds.stateless_random_map(fn)``).
  2. Subclassing and overriding ``stateless_random_map(self, element, key)``.

  Each element receives an immutable JAX PRNG key derived lazily via
  ``jax.random.key`` and ``jax.random.fold_in``.
  """

  def __init__(self, fn: Callable[[Any, Any], Any] | None = None):
    if fn is not None:
      if isinstance(fn, RandomMap) or not callable(fn):
        raise TypeError(
            "`JaxStatelessRandomMap` expects a callable"
            f" `transform(element, key)`, got {type(fn).__name__}."
        )
    self._wrapped_fn = fn

  def make_key_for_index(self, seed: int) -> Callable[[int], Any]:
    """Returns ``index -> key`` closure derived lazily via ``jax.random``."""
    # May raise ImportError if JAX is not linked.
    import jax  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    # Fold in `seed` once for dataset, then `index` for each element.
    # Note, `jax.random.fold_in` only consumes 32 bits of data, and 64-bit seeds
    # and indices are not preserved when `jax_enable_x64=False`. So here,
    # 64-bit seeds and indices are folded in as two uint32 words, low word
    # first. This is a Grain convention, not a JAX standard.
    seed_lo, seed_hi = _split_uint64(seed)
    base_key = jax.random.key(seed_lo)
    if seed_hi:
      # The high word is skipped when zero, so that values < 2**32 give exactly
      # `fold_in(key(seed), index)` at the cost of a single `fold_in` per
      # element.
      base_key = jax.random.fold_in(base_key, seed_hi)

    def _key_for_index(index: int) -> Any:
      lo, hi = _split_uint64(index)
      key = jax.random.fold_in(base_key, lo)
      if hi:
        key = jax.random.fold_in(key, hi)
      return key

    return _key_for_index

  def stateless_random_map(self, element: Any, key: Any) -> Any:
    """Applies the wrapped callable or subclass override with ``key``."""
    if self._wrapped_fn is None:
      raise NotImplementedError(
          "Subclasses of `JaxStatelessRandomMap` must override"
          " `stateless_random_map` or pass `fn` to `__init__`."
      )
    return self._wrapped_fn(element, key)


class Filter(abc.ABC):
  """Abstract base class for filter transformations for individual elements.

  The pipeline will drop any element for which the filter function returns
  False.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Example:
    Filtering out odd numbers from a dataset::

      import grain

      # Define a filter class that keeps only even numbers.
      class KeepEven(grain.transforms.Filter):
        def filter(self, element: int) -> bool:
          return element % 2 == 0

      # Create a parent dataset.
      parent_ds = grain.MapDataset.range(10)
      print(list(parent_ds))
      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      # Apply the filter transform.
      transformed_ds = parent_ds.filter(KeepEven())
      print(list(transformed_ds))
      # [0, 2, 4, 6, 8]
  """

  @abc.abstractmethod
  def filter(self, element) -> bool:
    """Filters a single element; returns True if the element should be kept.

    Args:
      element: The input element to evaluate.

    Returns:
      `True` if the element should be kept in the pipeline, `False` otherwise.
    """


class FlatMap(abc.ABC):
  """Abstract base class for splitting operations of individual elements.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Attributes
    max_fan_out: Absolute maximum number of splits that an element can generate.
    If element splits into number of sub-elements exceeding `max_fan_out`, an
    error is raised. In the case of variable fan-out, for performance reasons,
    please be mindful of the distribution in fan-outs. If the minimum and
    maximum fan-out in this distribution differ by several orders of magnitude,
    with a correspondingly very large `max_fan_out`, performance will degrade.
    In this case please consider preprocessing your data to keep the
    `max_fan_out` reasonable.
  """

  max_fan_out: int

  @abc.abstractmethod
  def flat_map(self, element) -> Sequence[Any]:
    """splits a single element."""


@dataclasses.dataclass(frozen=True)
class Batch:
  """Transformation that groups consecutive elements into batches.

  Refer to :func:`grain.MapDataset.batch` or :func:`grain.IterDataset.batch`
  documentation for more details.
  """
  batch_size: int
  drop_remainder: bool = False
  batch_fn: Callable[[Sequence[Any]], Any] | None = None


Transformation = Union[
    Batch,
    Map,
    RandomMap,
    StatelessRandomMap,
    TfRandomMap,
    Filter,
    FlatMap,
    MapWithIndex,
]
Transformations = Sequence[Transformation]


def get_pretty_transform_name(
    transform: Transformation | Callable[..., Any],
) -> str:
  """Returns a name for a transformation or callable with source file and line.

  Example: 'get_pretty_transform_name @ .../_src/core/transforms.py:116'

  Args:
    transform: The transfomation or callable to get the name of.
  """
  # We can't use `Transformation` here since `Union` does not support
  # `isinstance` check in Python 3.9.
  if isinstance(
      transform,
      (
          Batch,
          Map,
          RandomMap,
          StatelessRandomMap,
          TfRandomMap,
          Filter,
          FlatMap,
      ),
  ):
    # Check if transform class defines `__str__` and `__repr__` and use them if
    # so. Otherwise use the name of the class.
    if (
        transform.__class__.__str__ is object.__str__
        and transform.__class__.__repr__ is object.__repr__
    ):
      return transform.__class__.__name__
    return str(transform)

  # Some functions may not have `__name__`, e.g. `functools.partial`.
  transform_name = getattr(transform, "__name__", repr(transform))
  try:
    src_file = inspect.getsourcefile(transform)  # pyrefly: ignore[bad-argument-type]
    if src_file is None:
      return transform_name
    # If path is too long, shorten it to the last 3 parts.
    src_file_parts = src_file.split("/")
    if len(src_file_parts) >= 3:
      src_file = f".../{'/'.join(src_file_parts[-3:])}"
    src_lineno = inspect.getsourcelines(transform)[1]  # pyrefly: ignore[bad-argument-type]
    return f"{transform_name} @ {src_file}:{src_lineno}"
  except (OSError, TypeError):
    # `inspect` may raise if called on built-in functions.
    return transform_name
