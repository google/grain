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

import functools
import threading
from typing import Any, Callable, TypeVar

from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
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
  state["buffer"] = np.array([0, 0, 0, 0], dtype=np.uint64)
  state["buffer_pos"] = 4
  state["has_uint32"] = 0
  state["uinteger"] = 0
  rng.bit_generator.state = state


class RngPool:
  """RNG pool."""

  def __init__(self, seed: int):
    self._seed = seed
    self._generator_cache = []
    self._lock = threading.Lock()

  def __reduce__(self):
    return (RngPool, (self._seed,))

  def acquire_rng(self, index: int, *, op_seed: int = 0) -> np.random.Generator:
    """Acquire RNG."""
    with self._lock:
      if self._generator_cache:
        rng = self._generator_cache.pop()
      else:
        rng = np.random.Generator(np.random.Philox(self._seed))
    _reset_rng_state(rng, op_seed=op_seed, index=index)
    return rng

  def release_rng(self, rng: np.random.Generator):
    with self._lock:
      self._generator_cache.append(rng)


class MapMapDataset(dataset.MapDataset[T]):
  """Map transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.MapTransform | Callable[[Any], T],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.MapTransform):
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
      self._map_fn = transform.map
    else:
      self._map_fn = transform

  def __len__(self) -> int:
    return len(self._parent)

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __str__(self) -> str:
    return f"MapMapDataset(transform={self._transform_name})"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      if element is None:
        return None
      return self._stats.record_output_spec(self._map_fn(element))


class RandomMapMapDataset(dataset.MapDataset[T]):
  """Random map transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: (
          transforms.RandomMapTransform
          | Callable[[Any, np.random.Generator], T]
      ),
      seed: int | None = None,
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.RandomMapTransform):
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
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

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __str__(self) -> str:
    return f"RandomMapMapDataset(transform={self._transform_name})"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      if element is None:
        return None
      rng = self._rng_pool.acquire_rng(index)
      element = self._map_fn(element, rng)
      self._rng_pool.release_rng(rng)
      return self._stats.record_output_spec(element)


class MapWithIndexMapDataset(dataset.MapDataset[T]):
  """Map with index transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.MapWithIndex | Callable[[int, Any], T],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.MapWithIndex):
      self._map_fn = transform.map_with_index
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
    else:
      self._map_fn = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"MapWithIndexMapDataset(transform={self._transform_name})"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      if element is None:
        return None
      return self._stats.record_output_spec(self._map_fn(index, element))


class _MapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies map transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      map_fn: Callable[[Any], T],
      transform_name: str,
  ):
    super().__init__(parent)
    self._map_fn = map_fn
    self._transform_name = transform_name

  @stats.record_next_duration_if_output
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        element = self._map_fn(element)
      return self._stats.record_output_spec(element)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return f"MapDatasetIterator(transform={self._transform_name})"


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

  def __str__(self) -> str:
    return f"RandomMapDatasetIterator(transform={self._transform_name})"


class _MapWithIndexDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies map with index transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      map_fn: Callable[[int, Any], T],
      transform_name: str,
  ):
    super().__init__(parent)
    self._map_fn = map_fn
    self._transform_name = transform_name
    self._counter = 0

  @stats.record_next_duration_if_output
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        element = self._map_fn(self._counter, element)
      self._counter += 1
      return self._stats.record_output_spec(element)

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "counter": self._counter,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._counter = state["counter"]

  def __str__(self) -> str:
    return f"MapWithIndexDatasetIterator(transform={self._transform_name})"


class RandomMapIterDataset(dataset.IterDataset[T]):
  """Random map transformation for IterDataset."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: (
          transforms.RandomMapTransform
          | Callable[[Any, np.random.Generator], T]
      ),
      seed: int | None = None,
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.RandomMapTransform):
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
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

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __iter__(self) -> _RandomMapDatasetIterator[T]:
    return _RandomMapDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        seed=self._seed,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"RandomMapIterDataset(transform={self._transform_name})"


class MapIterDataset(dataset.IterDataset[T]):
  """Map transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.MapTransform | Callable[[Any], T],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.MapTransform):
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
      self._map_fn = transform.map
    else:
      self._map_fn = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __iter__(self) -> _MapDatasetIterator[T]:
    return _MapDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"MapIterDataset(transform={self._transform_name})"


class MapWithIndexIterDataset(dataset.IterDataset[T]):
  """Map with index transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.MapWithIndex | Callable[[int, Any], T],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.MapWithIndex):
      # Use the transform class name. The `cached_property` below will not
      # be called.
      self._transform_name = transform.__class__.__name__
      self._map_fn = transform.map_with_index
    else:
      self._map_fn = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __iter__(self) -> _MapWithIndexDatasetIterator[T]:
    return _MapWithIndexDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"MapWithIndexIterDataset(transform={self._transform_name})"
