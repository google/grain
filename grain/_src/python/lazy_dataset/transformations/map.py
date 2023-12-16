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
"""Map transformation for LazyDataset."""

import threading
from typing import Any, Callable, TypeVar, Union

from absl import logging
from grain._src.core import transforms
from grain._src.python.lazy_dataset import lazy_dataset
import numpy as np


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name

_MapTransformType = Union[
    transforms.MapTransform, transforms.RandomMapTransform, Callable[..., T]
]


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
  state["buffer"] = np.array([0, 0, 0, 0], dtype=np.uint64)
  state["buffer_pos"] = 4
  state["has_uint32"] = 0
  state["uint32"] = 0
  rng.bit_generator.state = state


class RngPool:
  """RNG pool."""

  def __init__(self, seed: int):
    self._seed = seed
    self._generator_cache = []
    self._lock = threading.Lock()

  def acquire_rng(self, index: int, *, op_seed: int = 0) -> np.random.Generator:
    """Acquire RNG."""
    if self._generator_cache:
      with self._lock:
        rng = self._generator_cache.pop()
    else:
      rng = np.random.Generator(np.random.Philox(self._seed))
    _reset_rng_state(rng, op_seed=op_seed, index=index)
    return rng

  def release_rng(self, rng: np.random.Generator):
    with self._lock:
      self._generator_cache.append(rng)


def _get_map_fn_and_seed(
    transform: _MapTransformType, seed: int | None = None
) -> tuple[Callable[..., T], int | None]:
  """Extracts a map fn from `transform`.

  If a seed is returned map fn requires a seed.

  Args:
    transform: A (random) map transform as object or callable.
    seed: Seed for random transform. Don't pass a seed if the transform is not
      random.

  Returns:
    Tuple of a callable and a seed. The callable expects the element to be
    mapped as first argument. If seed is not None the callable expects a
    second argument with a np.random.Generator.
  """
  if isinstance(transform, transforms.MapTransform):
    if seed is not None:
      logging.warning(
          "Provided seed for MapTransform %s which doesn't need a seed.",
          transform,
      )
    return transform.map, None
  elif isinstance(transform, transforms.RandomMapTransform):
    if seed is None:
      raise ValueError("RandomMapTransform requires random seed.")
    return transform.random_map, seed
  elif isinstance(transform, transforms.TfRandomMapTransform):
    if seed is None:
      raise ValueError("RandomMapTransform requires random seed.")
    return transform.np_random_map, seed
  else:
    # If a `seed` is provided we treat the Callable as RandomMapTransform
    return transform, seed


@lazy_dataset.lazy_map_dataset_function("map")
class MapLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Map LazyMapDataset."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      transform: _MapTransformType,
      seed: int | None = None,
  ):
    super().__init__(parent)
    self._map_fn, seed = _get_map_fn_and_seed(transform, seed)
    self._rng_pool = None if seed is None else RngPool(seed)

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    if element is None:
      return None
    if self._rng_pool:
      rng = self._rng_pool.acquire_rng(index)
      element = self._map_fn(element, rng)
      self._rng_pool.release_rng(rng)
    else:
      element = self._map_fn(element)
    return element


@lazy_dataset.lazy_map_dataset_function("map_with_index")
class MapWithIndexLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Map with index LazyMapDataset."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      transform: transforms.MapWithIndexTransform | Callable[[int, Any], T],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.MapWithIndexTransform):
      self._map_fn = transform.map_with_index
    else:
      # Expect Callable[[int, Any], T].
      self._map_fn = transform

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    if element is None:
      return None
    return self._map_fn(index, element)


class _MapLazyDatasetIterator(lazy_dataset.LazyDatasetIterator[T]):
  """Iterator that applies map transformation to elements."""

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
      map_fn: Callable[..., T],
      seed: int | None = None,
  ):
    super().__init__()
    self._parent = parent
    self._map_fn = map_fn
    self._index_for_rng = 0
    self._seed = seed
    self._rng = np.random.Generator(np.random.Philox(seed))

  def __next__(self):
    try:
      element = next(self._parent)
    except StopIteration as e:
      raise e

    if element is not None:
      if self._seed is not None:
        _reset_rng_state(self._rng, op_seed=0, index=self._index_for_rng)
        element = self._map_fn(element, self._rng)
      else:
        element = self._map_fn(element)

    self._index_for_rng += 1
    return element

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "index_for_rng": self._index_for_rng,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._index_for_rng = state["index_for_rng"]

  def __str__(self) -> str:
    return f"MapLazyDatasetIterator(parent={self._parent}"


@lazy_dataset.lazy_iter_dataset_function("map")
class MapLazyIterDataset(lazy_dataset.LazyIterDataset[T]):
  """Map transformation for LazyIterDatasets."""

  def __init__(
      self,
      parent: lazy_dataset.LazyIterDataset,
      transform: _MapTransformType,
      seed: int | None = None,
  ):
    super().__init__(parent)
    self._map_fn, self._seed = _get_map_fn_and_seed(transform, seed)

  def __iter__(self) -> _MapLazyDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _MapLazyDatasetIterator(
        parent_iter,
        map_fn=self._map_fn,
        seed=self._seed,
    )

  def __str__(self) -> str:
    return f"MapLazyIterDataset(parent={self._parent}"
