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

import functools
import os
import pickle
import shelve
import threading
from typing import Any, TypeVar

from grain._src.python.dataset import dataset


T = TypeVar("T")


class CacheMapDataset(dataset.MapDataset[T]):
  """Caches elements of a MapDataset in memory or to disk."""

  def __init__(self, parent: dataset.MapDataset[T], cache_path: str = ""):
    """Caches elements of a MapDataset.

    If cache_path is empty, elements are cached in memory. If cache_path is
    provided, elements are cached on disk in the given directory.

    Args:
      parent: The parent MapDataset whose elements are to be cached.
      cache_path: The directory to use for on-disk caching. If empty, in-memory
        caching is used.
    """
    super().__init__(parent)
    self._cache_path = cache_path
    self._cache_lock = threading.Lock()
    self._worker_index = 0

  def __getstate__(self):
    state = self.__dict__.copy()
    if "cache" in state:
      del state["cache"]
    state["_cache_lock"] = None
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._cache_lock = threading.Lock()

  @functools.cached_property
  def _cache(self) -> dict[str, Any] | shelve.Shelf:
    """Returns the cache object, initializing it if necessary."""
    if not self._cache_path:
      return {}

    # Per-process file caching in cache_path
    os.makedirs(self._cache_path, exist_ok=True)

    worker_id = f"worker_{self._worker_index}"
    cache_filename = os.path.join(self._cache_path, f"map_cache_{worker_id}")
    return shelve.open(cache_filename)

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)

    key = str(index)
    with self._cache_lock:
      if key in self._cache:
        return self._cache[key]

    value = self._parent[index]

    with self._cache_lock:
      # Check again in case another thread in same process filled it.
      if key not in self._cache:
        self._cache[key] = value
    return value

  def __del__(self):
    if "cache" in self.__dict__ and isinstance(self._cache, shelve.Shelf):
      self._cache.close()

  def __str__(self) -> str:
    return (
        f"CacheMapDataset(parent={self._parent},"
        f" cache_path='{self._cache_path}')"
    )

  def set_worker_index(self, worker_index: int):
    self._worker_index = worker_index


class CacheIterDataset(dataset.IterDataset[T]):
  """Caches elements of an IterDataset."""

  def __init__(self, parent: dataset.IterDataset[T], cache_path: str = ""):
    """Caches elements of an IterDataset.

    If cache_path is empty, elements are cached in memory. If cache_path is
    provided, elements are cached on disk in the given directory.

    Args:
      parent: The parent IterDataset whose elements are to be cached.
      cache_path: The directory to use for on-disk caching. If empty, in-memory
        caching is used.
    """
    super().__init__(parent)
    self._cache_path = cache_path

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _CacheDatasetIterator(self._parent.__iter__(), self._cache_path)

  def __str__(self) -> str:
    return (
        f"CacheIterDataset(parent={self._parent},"
        f" cache_path='{self._cache_path}')"
    )


class _CacheDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for CacheIterDataset."""

  def __init__(self, parent_iter: dataset.DatasetIterator[T], cache_path: str):
    super().__init__(parent_iter)
    self._cache_path = cache_path
    self._cache_filename: str | None = None
    self._mode = ""  # "mem", "read", "write"
    self._fp = None  # for file-based
    self._cache_filled = False  # for in-memory
    self._cache: list[T] = []  # for in-memory
    self._starting_state = parent_iter.get_state()  # for both
    self._pos = 0  # for both
    self._state_reset = False  # for both

  def _setup_mode(self):
    """Setups cache mode and opens file if necessary."""
    if self._cache_path:
      worker_id = f"worker_{self._ctx.mp_context.process_index}"
      self._cache_filename = self._cache_filename = os.path.join(
          self._cache_path, f"iter_cache_{worker_id}"
      )
    if self._mode:
      return
    if not self._cache_filename:
      self._mode = "mem"
      return

    try:
      self._fp = open(self._cache_filename, "rb")
      self._mode = "read"
    except FileNotFoundError:
      self._mode = "write"
      self._fp = None

  def _file_cleanup(self):
    if self._fp and not self._fp.closed:
      self._fp.close()
    if self._mode == "write":
      try:
        os.remove(self._cache_filename)
      except FileNotFoundError:
        pass

  def _next_mem(self) -> T:
    if self._pos < len(self._cache):
      val = self._cache[self._pos]
      self._pos += 1
      return val
    if self._cache_filled:
      raise StopIteration()
    if self._state_reset:
      self._state_reset = False
      while len(self._cache) < self._pos:
        self._cache.append(next(self._parent))
    try:
      val = next(self._parent)
      self._cache.append(val)
      self._pos += 1
      return val
    except StopIteration as exc:
      self._cache_filled = True
      raise StopIteration() from exc

  def _next_read(self) -> T:
    assert self._fp is not None
    if self._state_reset:
      self._state_reset = False
      self._fp.seek(0)  # Reset file pointer to start of file.
      for _ in range(self._pos):
        pickle.load(self._fp)
    try:
      val = pickle.load(self._fp)
      self._pos += 1
      return val
    except EOFError as exc:
      self._fp.close()
      raise StopIteration() from exc

  def _next_write(self) -> T:
    if self._state_reset:
      self._state_reset = False
      for _ in range(self._pos):
        val = next(self._parent)
        if self._fp is None:
          os.makedirs(self._cache_path, exist_ok=True)
          self._fp = open(self._cache_filename, "wb")
        pickle.dump(val, self._fp)
    try:
      val = next(self._parent)
      if self._fp is None:
        os.makedirs(self._cache_path, exist_ok=True)
        self._fp = open(self._cache_filename, "wb")
      pickle.dump(val, self._fp)
      self._pos += 1
      return val
    except StopIteration as exc:
      if self._fp is not None:
        self._fp.close()
      self._mode = ""  # force re-setup to reopen file for reading
      raise StopIteration() from exc

  def __next__(self) -> T:
    self._setup_mode()
    if self._mode == "mem":
      return self._next_mem()
    elif self._mode == "read":
      return self._next_read()
    elif self._mode == "write":
      return self._next_write()
    raise RuntimeError(f"Iterator in invalid state with mode {self._mode}.")

  def __del__(self):
    self._file_cleanup()

  def get_state(self) -> dict[str, Any]:
    return {"pos": self._pos}

  def set_state(self, state: dict[str, Any]):
    self._file_cleanup()
    self._parent.set_state(self._starting_state)
    if not self._cache_filled:
      self._cache = []
    self._pos = state["pos"]
    self._mode = ""  # force re-setup
    self._state_reset = True
