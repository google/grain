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
import functools
import sys
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
    # If the parent dataset is on infinite repeat, its length is
    # sys.maxsize and would result in overflows if further increased.
    # In this case, we just keep the length as sys.maxsize.
    if len(self._parent) >= sys.maxsize / self._transform.max_fan_out:
      return sys.maxsize
    return self._transform.max_fan_out * len(self._parent)

  def __str__(self) -> str:
    return (
        "FlatMapMapDataset("
        f"transform={transforms.get_pretty_transform_name(self._transform)})"
    )

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element_index, split_index = divmod(index, self._transform.max_fan_out)
    element = self._parent[element_index]
    with self._stats.record_self_time():
      if element is None:
        return None
      splits = self._transform.flat_map(element)
      if not isinstance(splits, Sequence):
        splits = list(splits)
      if len(splits) > self._transform.max_fan_out:
        raise ValueError(
            "The user-provided FlatMapTransform has a split that exceeds"
            " specified max fan-out size. To address this, you can raise the"
            " max fan-out size, but for a max fan-out size >100, performance"
            " may suffer. Please consider preprocessing your data to keep the"
            " max fan-out size reasonable."
        )
      if split_index < len(splits):
        return self._stats.record_output_spec(splits[split_index])
      return None


class _FlatMapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for flattening and mapping each element into many elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[S],
      flat_map: Callable[[S], Sequence[T]],
      transform_name: str,
  ):
    super().__init__(parent)
    self._flat_map = flat_map
    self._next_index_in_buffer = 0
    self._buffer = []
    self._last_parent_state = self._parent.get_state()
    self._transform_name = transform_name

  def _has_consumed_all_buffer_elements(self):
    return self._next_index_in_buffer >= len(self._buffer)

  @dataset_stats.record_next_duration_if_output
  def __next__(self):
    timer = dataset_stats.Timer()
    while self._has_consumed_all_buffer_elements():
      # Stores the previous state so that we can checkpoint this iterator
      # without storing `self._buffer` elements
      self._last_parent_state = self._parent.get_state()

      element = next(self._parent)  # Raises `StopIteration` when done.

      with timer:
        self._next_index_in_buffer = 0
        self._buffer = self._flat_map(element)

    with self._stats.record_self_time(offset_ns=timer.value()):
      mapped_element = self._buffer[self._next_index_in_buffer]
      self._next_index_in_buffer += 1
      return self._stats.record_output_spec(mapped_element)

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

  def __str__(self) -> str:
    return f"FlatMapDatasetIterator(transform={self._transform_name})"


class FlatMapIterDataset(dataset.IterDataset[T]):
  """Flat map for one-to-many split."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.FlatMapTransform,
  ):
    super().__init__(parent)
    self._transform = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._transform)

  def __str__(self) -> str:
    return f"FlatMapIterDataset(transform={self._transform_name})"

  def __iter__(self):
    parent_iter = self._parent.__iter__()
    return _FlatMapDatasetIterator(
        parent_iter, self._transform.flat_map, self._transform_name
    )
