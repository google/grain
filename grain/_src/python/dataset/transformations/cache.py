# Copyright 2025 Google LLC
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
"""Caching transformations."""

from __future__ import annotations

import os
from typing import Any, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats


T = TypeVar("T")


class CacheIterDataset(dataset.IterDataset[T]):
  """Caches elements of an IterDataset in memory."""

  def __init__(self, parent: dataset.IterDataset[T]):
    """Caches elements of an IterDataset in memory.

    Args:
      parent: The parent IterDataset whose elements are to be cached.
    """
    super().__init__(parent)

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _InMemoryCacheDatasetIterator(self._parent.__iter__())

  def __str__(self) -> str:
    return f"CacheIterDataset(parent={self._parent})"


class _InMemoryCacheDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for CacheIterDataset with in-memory caching."""

  def __init__(self, parent_iter: dataset.DatasetIterator[T]):
    super().__init__(parent_iter)
    self._cache: list[T] = []
    self._position = 0
    self._cache_filled = False
    self._rebuild_cache = False
    self._starting_state = parent_iter.get_state()

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    timer = stats.Timer()
    if self._cache_filled:
      with self._stats.record_self_time():
        if self._position < len(self._cache):
          val = self._cache[self._position]
          self._position += 1
          return self._stats.record_output_spec(val)
        else:
          raise StopIteration()
    if self._rebuild_cache:
      with timer:
        self._parent.set_state(self._starting_state)
        self._cache = []
        self._rebuild_cache = False
      for _ in range(self._position):
        val = next(self._parent)
        with timer:
          self._cache.append(val)
    try:
      val = next(self._parent)
      with self._stats.record_self_time(offset_ns=timer.value()):
        if len(self._cache) == self._position:
          self._cache.append(val)
        self._position += 1
        return self._stats.record_output_spec(val)
    except StopIteration as exc:
      self._cache_filled = True
      raise StopIteration() from exc

  def get_state(self) -> dict[str, Any]:
    return {
        "parent": self._parent.get_state(),
        "position": self._position,
        "cache_filled": self._cache_filled,
    }

  def set_state(self, state: dict[str, Any]):
    if state["cache_filled"] and not self._cache_filled:
      # If the state is from an iterator whose cache was filled, the parent
      # state will not be usable. We will rebuild the cache from scratch.
      self._rebuild_cache = True
    self._parent.set_state(state["parent"])
    self._position = state["position"]
    if self._position > len(self._cache) and not self._cache_filled:
      # If the position is greater than the cache size, we need to rebuild the
      # cache in order to fill in the missing elements.
      self._rebuild_cache = True
