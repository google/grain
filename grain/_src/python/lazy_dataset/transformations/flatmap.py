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
"""Flatmap transformation for LazyMapDataset."""

from typing import Any, TypeVar

from grain._src.core import transforms
from grain._src.python.lazy_dataset import lazy_dataset


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


class FlatMapLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Flat map for one-to-many split."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      transform: transforms.FlatMapTransform,
  ):
    super().__init__(parent)
    self._transform = transform

  def __len__(self) -> int:
    return self._transform.max_fan_out * len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    fan_out = self._transform.max_fan_out
    split_index = index % fan_out
    element_index = index // fan_out
    element = self._parent[element_index]
    splits = list(enumerate(self._transform.flat_map(element)))
    if len(splits) > fan_out:
      raise ValueError(
          "The user-provided FlatMapTransform has a split that exceeds"
          " specified max fan-out size. To address this, you can raise the max"
          " fan-out size, but for a max fan-out size >100, performance may"
          " suffer. Please consider preprocessing your data to keep the max"
          " fan-out size reasonable."
      )
    for i, sub_element in splits:
      if i == split_index:
        return sub_element
    return None
