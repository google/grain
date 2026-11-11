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
"""Random map transformation for LazyDataset."""

import functools
import threading
from typing import Any, Callable, Sequence, TypeVar

from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import map as map_dataset
import numpy as np

T = TypeVar("T")  # pylint: disable=invalid-name


# We need this little helper class to handle RNG generator for random map
# transformations. It manages a pool of RNG objects that can be re-used.
# Always creating new np.random.Philox objects seems expensive.
# We could probably do better by having our own RNG generator implemented in
# C++.


def _reset_rng_state(
    rng: np.random.Generator, op_seed: int, index: int
) -> None:
  state = rng.bit_generator.state
  state["state"]["counter"] = np.array([0, 0, op_seed, index], dtype=np.uint64)
  state["buffer"] = np.array([0, 0, 0, 0], dtype=np.uint64)  # pyrefly: ignore[unsupported-operation]
  state["buffer_pos"] = 4  # pyrefly: ignore[unsupported-operation]
  state["has_uint32"] = 0  # pyrefly: ignore[unsupported-operation]
  state["uinteger"] = 0  # pyrefly: ignore[unsupported-operation]
  rng.bit_generator.state = state


class RngPool:
  """Thread-safe pool of reusable per-index random number generators.

  Manages a cache of ``np.random.Generator`` (Philox) instances to avoid the
  overhead of constructing a new generator on every element access. Each
  acquired generator's state is deterministically reset based on the pool seed,
  element ``index``, and ``op_seed``.

  For standard element-wise random transformations, prefer
  ``MapDataset.random_map``, which manages an ``RngPool`` internally. Use
  ``grain.experimental.RngPool`` directly when implementing custom
  ``grain.MapDataset`` subclasses that require deterministic, index-based
  randomness inside ``__getitem__`` (such as randomly sampling other indices
  from the parent dataset to pair or mix with the current element).

  Example:
    Implementing a custom ``MapDataset`` that deterministically pairs each
    element with a randomly sampled other element from the parent dataset::

      import grain

      class ContrastiveMapDataset(grain.MapDataset):

        def __init__(self, parent: grain.MapDataset, seed: int):
          super().__init__(parent)
          self._rng_pool = grain.experimental.RngPool(seed)

        def __len__(self) -> int:
          return len(self._parent)

        def __getitem__(self, index):
          if isinstance(index, slice):
            return self.slice(index)
          rng = self._rng_pool.acquire_rng(index)
          other_index = int(rng.integers(0, len(self._parent) - 1))
          if other_index >= index:
            other_index += 1
          self._rng_pool.release_rng(rng)
          return (self._parent[index], self._parent[other_index])

      parent_ds = grain.MapDataset.source([10, 20, 30, 40, 50, 60, 70])
      paired_ds = ContrastiveMapDataset(parent_ds, seed=42)

      print(list(paired_ds))
      # [(10, 70), (20, 30), (30, 40), (40, 60), (50, 10), (60, 20), (70, 30)]

      # Accessing the same index always produces the same random pair.
      print(paired_ds[2])
      # (30, 40)
  """

  def __init__(self, seed: int):
    """Initializes the RngPool.

    Args:
      seed: Base seed used to initialize random number generators in the pool.
    """
    self._seed = seed
    self._generator_cache = []
    self._lock = threading.Lock()

  def __reduce__(self):
    return (RngPool, (self._seed,))

  def acquire_rng(self, index: int, *, op_seed: int = 0) -> np.random.Generator:
    """Acquires a random number generator with state reset for the given index.

    Reuses a cached generator from the pool if available, or creates a new
    Philox-backed ``np.random.Generator``, and resets its state using the pool
    seed, ``index``, and ``op_seed``.

    Args:
      index: Element index used to deterministically set the generator state.
      op_seed: Optional operation-specific seed used to distinguish multiple
        random operations at the same index.

    Returns:
      A ``np.random.Generator`` with deterministic state for ``(index,
      op_seed)``.
    """
    with self._lock:
      if self._generator_cache:
        rng = self._generator_cache.pop()
      else:
        rng = np.random.Generator(np.random.Philox(self._seed))
    _reset_rng_state(rng, op_seed=op_seed, index=index)
    return rng

  def release_rng(self, rng: np.random.Generator):
    """Returns a random number generator to the pool for reuse.

    Args:
      rng: The random number generator to return to the pool.
    """
    with self._lock:
      self._generator_cache.append(rng)


class RandomMapMapDataset(
    map_dataset._ElementSpecFromTransformMapDatasetMixin[T]  # pylint: disable=protected-access
):
  """Random map transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.RandomMap | Callable[[Any, np.random.Generator], T],
      seed: int | None = None,
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.RandomMap):
      self._map_fn = transform.random_map
    else:
      self._map_fn = transform
    seed = self._default_seed if seed is None else seed
    if seed is None:
      raise ValueError(
          "`random_map` requires a seed. Please either provide it with"
          " `ds.seed(seed)` before any random transformations or pass it"
          " directly with `ds.random_map(transform, seed=seed)`."
      )
    self._rng_pool = RngPool(seed)

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"RandomMapMapDataset(transform={self._transform_name})"

  def _random_map_element(self, element: Any, index: int) -> T:
    if element is None:
      return None  # pyrefly: ignore[bad-return]
    rng = self._rng_pool.acquire_rng(index)
    element = self._map_fn(element, rng)
    self._rng_pool.release_rng(rng)
    return element

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      mapped_element = self._random_map_element(element, index)
      return (
          self._stats.record_output_spec(mapped_element)
          if mapped_element is not None
          else None
      )

  def _getitems(self, indices: Sequence[int]):
    elements = self._parent._getitems(indices)  # pylint: disable=protected-access
    with self._stats.record_self_time(num_elements=len(indices)):
      processed_elements = [
          self._random_map_element(element, index)
          for element, index in zip(elements, indices)
      ]
    return self._stats.record_output_spec_for_batch(processed_elements)


class _RandomMapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies random map transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      map_fn: Callable[[Any, np.random.Generator], T],
      seed: int,
      transform_name: str,
  ):
    super().__init__(parent)
    self._map_fn = map_fn
    self._index_for_rng = 0
    self._seed = seed
    self._rng = np.random.Generator(np.random.Philox(seed))
    self._transform_name = transform_name

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        # Shift index for the current worker process in case of multiprocess
        # execution. The actual index value doesn't matter as long as it is
        # unique for each process.
        index_for_rng = (
            self._ctx.mp_context.process_index
            + self._index_for_rng * self._ctx.mp_context.process_count
        )
        _reset_rng_state(self._rng, op_seed=0, index=index_for_rng)
        element = self._map_fn(element, self._rng)

      self._index_for_rng += 1
      return self._stats.record_output_spec(element)

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "index_for_rng": self._index_for_rng,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._index_for_rng = state["index_for_rng"]

  def _get_next_index(self) -> int:
    return self._index_for_rng

  def _set_next_index(self, next_index: int):
    dataset.set_next_index(self._parent, next_index)
    self._index_for_rng = next_index

  def __str__(self) -> str:
    return f"RandomMapDatasetIterator(transform={self._transform_name})"


class RandomMapIterDataset(
    map_dataset._ElementSpecFromTransformIterDatasetMixin[T]  # pylint: disable=protected-access
):
  """Random map transformation for IterDataset."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.RandomMap | Callable[[Any, np.random.Generator], T],
      seed: int | None = None,
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.RandomMap):
      self._map_fn = transform.random_map
    else:
      self._map_fn = transform
    self._seed = self._default_seed if seed is None else seed
    if self._seed is None:
      raise ValueError(
          "`random_map` requires a seed. Please either provide it with"
          " `ds.seed(seed)` before any random transformations or pass it"
          " directly with `ds.random_map(transform, seed=seed)`."
      )

  def __iter__(self) -> _RandomMapDatasetIterator[T]:
    return _RandomMapDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        seed=self._seed,  # pyrefly: ignore[bad-argument-type]
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"RandomMapIterDataset(transform={self._transform_name})"


def _validate_stateless_seed(seed: int | None, name: str) -> int:
  """Validates a non-negative 64-bit integer seed for stateless PRNGs."""
  if seed is None:
    raise ValueError(
        f"`{name}` requires a seed. Please either provide it with"
        " `ds.seed(seed)` before any random transformations or pass it"
        f" directly with `{name}(..., seed=seed)`."
    )
  seed = int(seed)
  if not 0 <= seed < 2**64:
    raise ValueError(f"Seed must be in [0, 2**64), got {seed}.")
  return seed


def _unwrap_spec_target(transform: transforms.StatelessRandomMap) -> Any:
  wrapped_fn = getattr(transform, "_wrapped_fn", None)
  return wrapped_fn if wrapped_fn is not None else transform


def _as_stateless_random_map(
    transform: transforms.StatelessRandomMap | Callable[[Any, Any], Any],
    owner_name: str,
) -> transforms.StatelessRandomMap:
  """Normalizes a `StatelessRandomMap` or callable (defaulting to JAX)."""
  if isinstance(transform, transforms.StatelessRandomMap):
    return transform
  if isinstance(transform, transforms.RandomMap) or not callable(transform):
    raise TypeError(
        f"`{owner_name}` expects a callable `transform(element, key)` or"
        f" `StatelessRandomMap`, got {type(transform).__name__}."
    )
  return transforms.JaxStatelessRandomMap(transform)


class StatelessRandomMapMapDataset(
    map_dataset._ElementSpecFromTransformMapDatasetMixin[T]  # pylint: disable=protected-access
):
  """Stateless key-based random map transformation for ``MapDataset``."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.StatelessRandomMap | Callable[[Any, Any], T],
      seed: int | None = None,
  ):
    stateless_transform = _as_stateless_random_map(
        transform, "stateless_random_map"
    )
    super().__init__(parent, _unwrap_spec_target(stateless_transform))
    self._stateless_transform = stateless_transform
    resolved_seed = self._default_seed if seed is None else seed
    self._seed = _validate_stateless_seed(resolved_seed, "stateless_random_map")

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    state.pop("_key_for_index", None)
    return state

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"StatelessRandomMapMapDataset(transform={self._transform_name})"

  @functools.cached_property
  def _key_for_index(self) -> Callable[[int], Any]:
    return self._stateless_transform.make_key_for_index(self._seed)

  def _random_map_element(self, element: Any, index: int) -> T | None:
    if element is None:
      return None
    return self._stateless_transform.stateless_random_map(
        element, self._key_for_index(index)
    )

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      mapped_element = self._random_map_element(element, index)
      return (
          self._stats.record_output_spec(mapped_element)
          if mapped_element is not None
          else None
      )

  def _getitems(self, indices: Sequence[int]):
    elements = self._parent._getitems(indices)  # pylint: disable=protected-access
    with self._stats.record_self_time(num_elements=len(indices)):
      processed_elements = [
          self._random_map_element(element, index)
          for element, index in zip(elements, indices)
      ]
    return self._stats.record_output_spec_for_batch(processed_elements)


class _StatelessRandomMapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies a stateless random map transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      stateless_transform: transforms.StatelessRandomMap,
      seed: int,
      transform_name: str,
  ):
    super().__init__(parent)
    self._stateless_transform = stateless_transform
    self._index_for_rng = 0
    self._seed = seed
    self._transform_name = transform_name

  @functools.cached_property
  def _key_for_index(self) -> Callable[[int], Any]:
    return self._stateless_transform.make_key_for_index(self._seed)

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        index_for_rng = (
            self._ctx.mp_context.process_index
            + self._index_for_rng * self._ctx.mp_context.process_count
        )
        element = self._stateless_transform.stateless_random_map(
            element, self._key_for_index(index_for_rng)
        )
      self._index_for_rng += 1
      return self._stats.record_output_spec(element)

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "index_for_rng": self._index_for_rng,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._index_for_rng = state["index_for_rng"]

  def _get_next_index(self) -> int:
    return self._index_for_rng

  def _set_next_index(self, next_index: int):
    dataset.set_next_index(self._parent, next_index)
    self._index_for_rng = next_index

  def __str__(self) -> str:
    return (
        f"StatelessRandomMapDatasetIterator(transform={self._transform_name})"
    )


class StatelessRandomMapIterDataset(
    map_dataset._ElementSpecFromTransformIterDatasetMixin[T]  # pylint: disable=protected-access
):
  """Stateless key-based random map transformation for ``IterDataset``."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.StatelessRandomMap | Callable[[Any, Any], T],
      seed: int | None = None,
  ):
    stateless_transform = _as_stateless_random_map(
        transform, "stateless_random_map"
    )
    super().__init__(parent, _unwrap_spec_target(stateless_transform))
    self._stateless_transform = stateless_transform
    resolved_seed = self._default_seed if seed is None else seed
    self._seed = _validate_stateless_seed(resolved_seed, "stateless_random_map")

  def __iter__(self) -> _StatelessRandomMapDatasetIterator[T]:
    return _StatelessRandomMapDatasetIterator(
        self._parent.__iter__(),
        stateless_transform=self._stateless_transform,
        seed=self._seed,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"StatelessRandomMapIterDataset(transform={self._transform_name})"
