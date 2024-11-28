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
"""Implements global shuffle transformation."""

from __future__ import annotations

from typing import TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.experimental.index_shuffle.python import index_shuffle_module as index_shuffle


T = TypeVar("T")


class ShuffleMapDataset(dataset.MapDataset[T]):
  """Shuffles the parent dataset."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      *,
      seed: int | None = None,
  ):
    super().__init__(parent)
    seed = self._default_seed if seed is None else seed
    if seed is None:
      raise ValueError(
          "`shuffle` requires a seed. Please provide it with `ds.seed(seed)`"
      )
    if seed < 0 or seed >= 2**32:
      raise ValueError(
          f"Seed must be an integer between 0 and 2**32-1 (got {seed=})."
      )
    self._seed = int(seed)

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return "ShuffleMapDataset"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      length = len(self._parent)
      epoch, index_in_epoch = divmod(index, length)
      # Note:
      #   - index_shuffle expects 32-bit integers
      #   - we use different seeds for each epoch to ensure that the shuffle is
      #     different for each epoch
      per_epoch_seed = (self._seed + epoch) % 2**32
      # raise RuntimeError(f"Expected int for index_in_epoch, got {type(index_in_epoch)}; Expected int for length - 1, got {type(length - 1)}; Expected int for per_epoch_seed, got {type(per_epoch_seed)}")
      shuffled_index_in_epoch = index_shuffle.index_shuffle(
          index=index_in_epoch, max_index=length - 1, seed=per_epoch_seed, rounds=4
      )
      shuffled_index = shuffled_index_in_epoch + epoch * length
    return self._parent[shuffled_index]


class WindowShuffleMapDataset(dataset.MapDataset[T]):
  """Shuffles the parent dataset within a given window.

  Shuffles the retrieval index within a range, given by window_size. Each unique
  index corresponds to exactly one shuffled index (i.e. there is a one-to-one
  mapping and hence a guarantee that no shuffled indices are repeated within a
  given window).
  """

  def __init__(
      self, parent: dataset.MapDataset, *, window_size: int, seed: int
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return "WindowShuffleMapDataset"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      window_index, index_in_window = divmod(index, self._window_size)
      seed = self._seed + window_index
      index_in_window = index_shuffle.index_shuffle(
          index_in_window,
          self._window_size - 1,
          seed,
          4,
      )
      index = index_in_window + window_index * self._window_size
    return self._stats.record_output_spec(self._parent[index])


class WindowShuffleIterDataset(dataset.IterDataset[T]):
  """Shuffles the parent dataset within a given window.

  Shuffles the iterator's next within a range, given by window_size. There is a
  one-to-one mapping and hence a guarantee that no shuffled indices are repeated
  within a given window).
  """

  def __init__(
      self, parent: dataset.IterDataset, *, window_size: int, seed: int
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed

  def __iter__(self) -> _WindowShuffleDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _WindowShuffleDatasetIterator(
        parent_iter, window_size=self._window_size, seed=self._seed
    )

  def __str__(self) -> str:
    return "WindowShuffleIterDataset"


class _WindowShuffleDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that within awindow shuffles elements."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      window_size: int,
      seed: int,
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed
    self._len = 0
    self._windows_iterators: list[dict[str, dataset.DatasetIterator[T]]] = []
    self._iter_pos = 0

    while True:
      try:
        next(parent)
        if self._len % window_size == 0:
          self._windows_iterators.append(self._parent.__iter__().get_state())
        self._len += 1
      except StopIteration:
        break

  def __next__(self):
    index_window, index_in_window = divmod(self._iter_pos, self._window_size)
    self._iter_pos += 1
    seed = self._seed + index_window
    index_in_window = index_shuffle.index_shuffle(
        index_in_window,
        max_index=self._window_size - 1,
        seed=seed,
        rounds=4,
    )
    tmp_iter: dataset.DatasetIterator[T] = self._parent.__iter__()
    tmp_iter.set_state(self._windows_iterators[index_window])
    # Due to iterator semantics, next() points to the next position but we need
    # the current position.
    value = next(tmp_iter) - 1
    for _ in range(index_in_window - 1):
      value = next(tmp_iter)
    return value

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return "WindowShuffleDatasetIterator"
