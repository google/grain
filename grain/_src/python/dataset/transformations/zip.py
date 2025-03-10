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

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats

T = TypeVar("T")


class ZipMapDataset(dataset.MapDataset[T]):
  """Combines MapDatasets of the same length to return a tuple of items."""

  def __init__(self, parents: Iterable[dataset.MapDataset[T]]):
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
    return f"ZipMapDataset(parents={self._parents}"


class ZipIterDataset(dataset.IterDataset[T]):
  """Combines IterDatasets of the same length to return a tuple of items."""

  def __init__(self, parents: Iterable[dataset.IterDataset[T]]):
    super().__init__([p for p in parents])
    if not self._parents:
      raise ValueError("At least one parent must be provided.")

  def __iter__(self) -> ZipDatasetIterator[T]:
    return ZipDatasetIterator(self._parents)

  def __str__(self) -> str:
    return f"ZipIterDataset(parents={self._parents})"


class ZipDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for ZipIterDataset."""

  def __init__(self, parents: Iterable[dataset.IterDataset[T]]):
    super().__init__([p.__iter__() for p in parents])

  @dataset_stats.record_next_duration_if_output
  def __next__(self) -> tuple[T, ...]:
    with self._stats.record_self_time():
      # Can't use for a `for` loop because we need to raise StopIteration from
      # the inner iterators.
      items = []
      i = 0
      while i < len(self._parents):
        it = self._parents[i]
        items.append(next(it))
        i += 1
      return self._stats.record_output_spec(tuple(items))

  def get_state(self) -> dict[str, Any]:
    return {"parents": [it.get_state() for it in self._parents]}

  def set_state(self, state):
    for it, s in zip(self._parents, state["parents"]):
      it.set_state(s)

  def __str__(self) -> str:
    return f"ZipDatasetIterator([{len(self._parents)} parents])"
