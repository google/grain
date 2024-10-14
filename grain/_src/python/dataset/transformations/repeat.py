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
import sys
from typing import Optional, TypeVar

from grain._src.python.dataset import dataset

T = TypeVar("T")


class RepeatMapDataset(dataset.MapDataset[T]):
  """Repeats the underlying dataset for num_epochs.

  This effectively just changes the length, which indicates the size of a single
  epoch, of the dataset. This makes it easier to iterate for a fixed number
  of steps.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      num_epochs: Optional[int] = None,
  ):
    super().__init__(parent)
    if num_epochs is not None and num_epochs <= 0:
      raise ValueError(f"num_epochs must be positive, but got {num_epochs}.")
    if len(parent) >= sys.maxsize:
      raise ValueError(
          f"Repeating already infinite dataset {parent} does nothing."
      )
    self._num_epochs = num_epochs
    if num_epochs is None:
      if len(parent) == 0:  # pylint: disable=g-explicit-length-test
        self._length: int = 0
      else:
        self._length: int = sys.maxsize
    else:
      self._length = num_epochs * len(parent)

  def __len__(self) -> int:
    return self._length

  def __str__(self) -> str:
    return f"RepeatMapDataset(num_epochs={self._num_epochs})"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    return self._stats.record_output_spec(self._parent[index])
