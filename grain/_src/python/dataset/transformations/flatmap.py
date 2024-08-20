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
"""Flatmap transformation for MapDataset."""

from typing import Any, Callable, Sequence, TypeVar

from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats


Element = Any
T = TypeVar("T")
S = TypeVar("S")


class FlatMapMapDataset(dataset.MapDataset[T]):
  """Flat map for one-to-many split."""

  def __init__(
      self,
      parent: dataset.MapDataset,
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


class _FlatMapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for flattening and mapping each element into many elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[S],
      flat_map: Callable[[S], Sequence[T]],
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._flat_map = flat_map
    self._next_index_in_buffer = 0
    self._buffer = []
    self._last_parent_state = self._parent.get_state()

  def _has_consumed_all_buffer_elements(self):
    return self._next_index_in_buffer >= len(self._buffer)

  def __next__(self):
    while self._has_consumed_all_buffer_elements():
      # Stores the previous state so that we can checkpoint this iterator
      # without storing `self._buffer` elements
      self._last_parent_state = self._parent.get_state()

      element = next(self._parent)  # Raises `StopIteration` when done.

      self._next_index_in_buffer = 0
      self._buffer = self._flat_map(element)

    mapped_element = self._buffer[self._next_index_in_buffer]
    self._next_index_in_buffer += 1
    return mapped_element

  def get_state(self):
    return {
        "parent": self._last_parent_state,
        "next_index_in_buffer": self._next_index_in_buffer,
    }

  def set_state(self, state: dict[str, Any]):
    self._last_parent_state = state["parent"]
    self._parent.set_state(state["parent"])
    self._next_index_in_buffer = state["next_index_in_buffer"]
    try:
      element = next(self._parent)
      # Recovers the buffer
      self._buffer = self._flat_map(element)
    except StopIteration:
      # Edge case: The iterator has run out of elements.
      # We keep this EOF state.

      self._next_index_in_buffer = 0
      pass


class FlatMapIterDataset(dataset.IterDataset[T]):
  """Flat map for one-to-many split."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.FlatMapTransform,
  ):
    super().__init__(parent)
    self._transform = transform

  def __iter__(self):
    parent_iter = self._parent.__iter__()
    return _FlatMapDatasetIterator(
        parent_iter, self._transform.flat_map, self._stats
    )
