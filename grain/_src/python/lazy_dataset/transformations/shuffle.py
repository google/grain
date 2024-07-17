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

from typing import TypeVar

from grain._src.python.experimental.index_shuffle.python import index_shuffle_module as index_shuffle
from grain._src.python.lazy_dataset import lazy_dataset


T = TypeVar("T")


class ShuffleMapDataset(lazy_dataset.MapDataset[T]):
  """Shuffles the parent dataset."""

  def __init__(
      self,
      parent: lazy_dataset.MapDataset[T],
      *,
      seed: int,
  ):
    super().__init__(parent)
    if seed < 0 or seed >= 2**32:
      raise ValueError(
          f"Seed must be an integer between 0 and 2**32-1 (got {seed=})."
      )
    self._seed = seed

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    length = len(self._parent)
    epoch, index_in_epoch = divmod(index, length)
    # Note:
    #   - index_shuffle expects 32-bit integers
    #   - we use different seeds for each epoch to ensure that the shuffle is
    #     different for each epoch
    per_epoch_seed = (self._seed + epoch) % 2**32
    shuffled_index_in_epoch = index_shuffle.index_shuffle(
        index_in_epoch, max_index=length - 1, seed=per_epoch_seed, rounds=4
    )
    shuffled_index = shuffled_index_in_epoch + epoch * length
    return self._parent[shuffled_index]


class WindowShuffleMapDataset(lazy_dataset.MapDataset[T]):
  """Shuffles the parent dataset within a given window.

  Shuffles the retrieval index within a range, given by window_size. Each unique
  index corresponds to exactly one shuffled index (i.e. there is a one-to-one
  mapping and hence a guarantee that no shuffled indices are repeated within a
  given window).
  """

  def __init__(
      self, parent: lazy_dataset.MapDataset, *, window_size: int, seed: int
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    window_index, index_in_window = divmod(index, self._window_size)
    seed = self._seed + window_index
    index_in_window = index_shuffle.index_shuffle(
        index_in_window,
        max_index=self._window_size - 1,
        seed=seed,
        rounds=4,
    )
    index = index_in_window + window_index * self._window_size
    return self._parent[index]
