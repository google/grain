# Copyright 2024 Google LLC
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
"""Implements zip transformation."""
from collections.abc import Iterable
from typing import TypeVar

from grain._src.python.lazy_dataset import lazy_dataset

T = TypeVar("T")


class ZipLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Combines LazyMapDatasets of the same length to return a tuple of items."""

  def __init__(self, parents: Iterable[lazy_dataset.LazyMapDataset[T]]):
    super().__init__([p for p in parents])
    lengths = [len(p) for p in self._parents]
    if not lengths:
      raise ValueError("At least one parent must be provided.")
    if not all(lengths[0] == l for l in lengths):
      raise ValueError("All parents must have the same length.")
    self._length = lengths[0]

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    return tuple(p[index] for p in self._parents)

  def __str__(self) -> str:
    return f"ZipLazyMapDataset(parents={self._parents}"
