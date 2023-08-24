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
"""Implements repeat transformation."""
import dataclasses
import sys
from typing import TypeVar

from grain._src.python.lazy_dataset import lazy_dataset

T = TypeVar("T")


@lazy_dataset.lazy_map_dataset_function("repeat")
@dataclasses.dataclass
class RepeatLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Repeats the underlying dataset for num_epochs.

  This effectively just changes the length, which indicates the size of a single
  epoch, of the dataset. This makes it easier to iterate for a fixed number
  of steps.
  """

  parent: lazy_dataset.LazyMapDataset[T]
  num_epochs: int | None
  _len: int = sys.maxsize

  def __post_init__(self):
    if len(self.parent) >= sys.maxsize:
      raise ValueError(
          f"Repeating already infinite dataset {self.parent} does nothing."
      )
    if self.num_epochs is None:
      self._len = sys.maxsize
    else:
      self._len = self.num_epochs * len(self.parent)

  @property
  def sparse(self) -> bool:
    return self.parent.sparse

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, index: int) -> T:
    return self.parent[index]
