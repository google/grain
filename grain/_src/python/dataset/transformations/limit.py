# Copyright 2025 Google LLC
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
"""Implements limit transformations."""

from typing import Any, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats

Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


class _LimitDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that limits the number of elements in the dataset."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[T],
      count: int,
  ):
    super().__init__(parent)
    self._count = count
    self._count_elements_read = 0

  @stats.record_next_duration_if_output
  def __next__(self):
    if self._count_elements_read >= self._count:
      raise StopIteration
    value = next(self._parent)
    self._count_elements_read += 1
    return value

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "count_elements_read": self._count_elements_read,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._count_elements_read = state["count_elements_read"]


class LimitIterDataset(dataset.IterDataset[T]):
  """Limits the number of elements in the dataset.

  Example usage:

  ```
  list(LimitIterDataset(MapDataset.range(5).to_iter_dataset(), 2) == [0, 1]
  ```

  Attributes:
    parent: The dataset to limit.
    count: The maximum number of elements to include in the dataset.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      count: int,
  ):
    """Initializes the limit dataset."""
    if count <= 0:
      raise ValueError(f"Count must be a non-negative integer. Got {count}")
    super().__init__(parent)
    self._count = count

  def __iter__(self) -> _LimitDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _LimitDatasetIterator(parent_iter, self._count)

  def __str__(self) -> str:
    return f"LimitIterDataset(count={self._count})"
