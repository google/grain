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

import bagz
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
import pickle


T = TypeVar("T")


class CacheIterDataset(dataset.IterDataset[T]):
  """Caches elements of an IterDataset."""

  def __init__(self, parent: dataset.IterDataset[T], cache_path: str = ""):
    """Caches elements of an IterDataset.

    If cache_path is None, elements are cached in memory. If cache_path is
    provided, elements are cached on disk in the given directory.

    The format of the cache files is considered an implementation detail and we
    do not guarantee backwards compatibility. Users should not rely on these
    files and only consume them through `CacheIterDataset`.

    Args:
      parent: The parent IterDataset whose elements are to be cached.
      cache_path: The directory to use for on-disk caching. If empty, in-memory
        caching is used.
    """
    super().__init__(parent)
    self._cache_path = cache_path

  def __iter__(self) -> dataset.DatasetIterator[T]:
    if not self._cache_path:
      return _InMemoryCacheDatasetIterator(self._parent.__iter__())
    return _FileCacheDatasetIterator(
        self._parent.__iter__(), self._cache_path
    )

  def __str__(self) -> str:
    return (
        f"CacheIterDataset(parent={self._parent},"
        f" cache_path='{self._cache_path}')"
    )


class _InMemoryCacheDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for CacheIterDataset with in-memory caching."""

  def __init__(self, parent_iter: dataset.DatasetIterator[T]):
    super().__init__(parent_iter)
    self._cache_filled = False
    self._cache: list[T] = []
    self._starting_state = parent_iter.get_state()
    self._pos = 0
    self._state_reset = False

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    timer = stats.Timer()
    if self._pos < len(self._cache):
      with self._stats.record_self_time():
        val = self._cache[self._pos]
        self._pos += 1
        return self._stats.record_output_spec(val)
    if self._cache_filled:
      raise StopIteration()
    if self._state_reset:
      # Rebuild the cache if state was reset before the cache was filled.
      self._state_reset = False
      while len(self._cache) < self._pos:
        val = next(self._parent)
        with timer:
          self._cache.append(val)
    try:
      val = next(self._parent)
      with self._stats.record_self_time(offset_ns=timer.value()):
        self._cache.append(val)
        self._pos += 1
        return self._stats.record_output_spec(val)
    except StopIteration as exc:
      self._cache_filled = True
      raise StopIteration() from exc

  def get_state(self) -> dict[str, Any]:
    return {"pos": self._pos}

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(self._starting_state)
    if not self._cache_filled:
      self._cache = []
    self._pos = state["pos"]
    self._state_reset = True


class _FileCacheDatasetIteratorBase(dataset.DatasetIterator[T]):
  """Base iterator for CacheIterDataset with file-based caching."""

  def __init__(self, parent_iter: dataset.DatasetIterator[T], cache_path: str):
    super().__init__(parent_iter)
    self._cache_path = cache_path
    self._cache_filename: str | None = None
    self._bag_writer: bagz.BagWriter | None = None
    self._bag_reader: bagz.BagReader | None = None
    self._starting_state = parent_iter.get_state()
    self._pos = 0
    self._state_reset = False

  def _exists(self, path: str) -> bool:
    raise NotImplementedError

  def _remove(self, path: str) -> None:
    raise NotImplementedError

  def _makedirs(self, path: str) -> None:
    raise NotImplementedError

  def _serialize(self, value: T) -> bytes:
    raise NotImplementedError

  def _deserialize(self, value_bytes: bytes) -> T:
    raise NotImplementedError

  def _file_cleanup(self):
    if self._bag_writer:
      self._bag_writer.close()
    if self._bag_writer is not None:
      self._remove(self._cache_filename)
    self._bag_writer = None
    self._bag_reader = None

  def _next_read(self, timer: stats.Timer) -> T:
    with self._stats.record_self_time(offset_ns=timer.value()):
      if self._state_reset:
        self._state_reset = False
      try:
        val_bytes = self._bag_reader[self._pos]
        val = self._deserialize(val_bytes)
        self._pos += 1
        return val
      except IndexError as exc:
        raise StopIteration() from exc

  def _next_write(self, timer: stats.Timer) -> T:
    if self._state_reset:
      self._state_reset = False
      for _ in range(self._pos):
        val = next(self._parent)
        with timer:
          if self._bag_writer is None:
            self._makedirs(self._cache_path)
            self._bag_writer = bagz.BagWriter(self._cache_filename)
          self._bag_writer.write(self._serialize(val))
    try:
      val = next(self._parent)
      with self._stats.record_self_time(offset_ns=timer.value()):
        if self._bag_writer is None:
          self._makedirs(self._cache_path)
          self._bag_writer = bagz.BagWriter(self._cache_filename)
        self._bag_writer.write(self._serialize(val))
        self._pos += 1
        return val
    except StopIteration as exc:
      if self._bag_writer is not None:
        self._bag_writer.close()
        self._bag_writer = None
      raise StopIteration() from exc

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    timer = stats.Timer()
    with timer:
      if self._state_reset:
        self._file_cleanup()
      if not self._cache_filename:
        worker_id = f"worker_{self._ctx.mp_context.process_index}"
        self._cache_filename = os.path.join(
            self._cache_path, f"iter_cache_{worker_id}.bagz"
        )
      if self._bag_reader is None and self._bag_writer is None:
        if self._exists(self._cache_filename):
          self._bag_reader = bagz.BagReader(self._cache_filename)
    if self._bag_reader is not None:
      return self._stats.record_output_spec(self._next_read(timer))
    else:
      return self._stats.record_output_spec(self._next_write(timer))

  def __del__(self):
    self._file_cleanup()

  def get_state(self) -> dict[str, Any]:
    return {"pos": self._pos}

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(self._starting_state)
    self._pos = state["pos"]
    self._state_reset = True


class _FileCacheDatasetIterator(_FileCacheDatasetIteratorBase[T]):
  """Iterator for CacheIterDataset with file-based caching."""

  def _exists(self, path: str) -> bool:
    return os.path.exists(path)

  def _remove(self, path: str) -> None:
    try:
      os.remove(path)
    except FileNotFoundError:
      pass

  def _makedirs(self, path: str) -> None:
    os.makedirs(path, exist_ok=True)

  def _serialize(self, value: T) -> bytes:
    return pickle.dumps(value)

  def _deserialize(self, value_bytes: bytes) -> T:
    return pickle.loads(value_bytes)
