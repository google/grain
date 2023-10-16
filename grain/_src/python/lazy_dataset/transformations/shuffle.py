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


@lazy_dataset.lazy_map_dataset_function("shuffle")
class ShuffleLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Shuffles the parent dataset."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset[T],
      *,
      reshuffle_each_epoch: bool = True,
      seed: int,
  ):
    super().__init__(parent)
    self._seed = seed
    self._reshuffle_each_epoch = reshuffle_each_epoch

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    length = len(self._parent)
    epoch = index // length
    index_in_epoch = index % length
    if self._reshuffle_each_epoch:
      # index_shuffle expects 32-bit integers
      seed = (self._seed + epoch) % 2**32
    else:
      seed = self._seed
    shuffled_index_in_epoch = index_shuffle.index_shuffle(
        index_in_epoch, max_index=length - 1, seed=seed, rounds=4
    )
    shuffled_index = shuffled_index_in_epoch + epoch * length
    return self._parent[shuffled_index]
