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
"""Implements slice transformation."""
from typing import TypeVar

from grain._src.python.lazy_dataset import lazy_dataset

T = TypeVar("T")


@lazy_dataset.lazy_map_dataset_function("slice")
class SliceLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Slices a LazyMapDataset similar to the slicing syntax in Python."""

  def __init__(self, parent: lazy_dataset.LazyMapDataset[T], sl: slice):
    super().__init__(parent)
    if not isinstance(sl, slice):
      raise ValueError(f"sl is not a slice: {type(sl)}")
    self._start, self._stop, self._step = sl.indices(len(parent))
    self._length = len(range(self._start, self._stop, self._step))

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return SliceLazyMapDataset(self, index)
    new_index = self._start + (index % self._length) * self._step
    return self._parent[new_index]

  def __str__(self) -> str:
    return f"{self._parent}[{self._start}:{self._stop}:{self._step}]"
