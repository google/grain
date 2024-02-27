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
"""Implements shard transformation."""

import dataclasses
from typing import TypeVar

from grain._src.core import sharding
from grain._src.python.lazy_dataset import lazy_dataset

T = TypeVar("T")


@lazy_dataset.lazy_map_dataset_function("shard")
@dataclasses.dataclass(frozen=False)
class ShardLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Shards a LazyMapDataset similar to the slicing syntax in Python."""

  parent: lazy_dataset.LazyMapDataset[T]
  shard_options: sharding.ShardOptions

  def __post_init__(self):
    self._start, self._end = sharding.even_split(
        len(self.parent), self.shard_options
    )

  @property
  def sparse(self) -> bool:
    return self.parent.sparse

  def __len__(self) -> int:
    return self._end - self._start  # pytype: disable=unsupported-operands

  def __getitem__(self, index: int) -> T | None:
    epoch = index // len(self)
    index_in_epoch = index % len(self)
    index = epoch * len(self.parent) + index_in_epoch + self._start  # pytype: disable=unsupported-operands
    return self.parent[index]
