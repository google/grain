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
from typing import Optional, Sequence, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats

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
      *,
      reseed_each_epoch: bool = True,
  ):
    super().__init__(parent)
    if num_epochs is not None and num_epochs <= 0:
      raise ValueError(f"num_epochs must be positive, but got {num_epochs}.")
    if len(parent) >= sys.maxsize:
      raise ValueError(
          f"Repeating already infinite dataset {parent} does nothing."
      )
    self._num_epochs = num_epochs
    self._parent_length = len(parent)
    if num_epochs is None:
      if self._parent_length == 0:  # pylint: disable=g-explicit-length-test
        self._length: int = 0
      else:
        self._length: int = sys.maxsize
    else:
      self._length = num_epochs * self._parent_length
    self._reseed_each_epoch = reseed_each_epoch

  def __len__(self) -> int:
    return self._length

  def __str__(self) -> str:
    return f"RepeatMapDataset(num_epochs={self._num_epochs})"

  def _getitems(self, indices: Sequence[int]):
    if not self._reseed_each_epoch:
      indices = [index % self._parent_length for index in indices]
    return self._parent._getitems(indices)  # pylint: disable=protected-access

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    if not self._reseed_each_epoch:
      # Use elements from the first epoch.
      index = index % self._parent_length
    return self._stats.record_output_spec(self._parent[index])


class _RepeatDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that repeats elements from parent iterator."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.DatasetIterator[T],
      num_epochs: int | None,
  ):
    super().__init__(parent)
    self._num_epochs = num_epochs
    self._epoch = 0
    self._parent_starting_state = self._parent.get_state()

  @stats.record_next_duration_if_output
  def __next__(self):
    timer = stats.Timer()
    if self._epoch == self._num_epochs:
      raise StopIteration
    while True:
      try:
        elem = next(self._parent)
        with self._stats.record_self_time(offset_ns=timer.value()):
          return self._stats.record_output_spec(elem)
      except StopIteration as exc:
        with timer:
          self._epoch += 1
          if self._num_epochs is not None and self._epoch == self._num_epochs:
            raise StopIteration from exc
          else:
            self._parent.set_state(self._parent_starting_state)

  def get_state(self):
    return {"parent": self._parent.get_state(), "epoch": self._epoch}

  def set_state(self, state):
    self._epoch = state["epoch"]
    self._parent.set_state(state["parent"])

  def __str__(self) -> str:
    return f"_RepeatDatasetIterator(num_epochs={self._num_epochs})"


class RepeatIterDataset(dataset.IterDataset[T]):
  """Repeats the underlying dataset for num_epochs.

  If num_epochs is None, repeats indefinitely.
  Note that unlike RepeatMapDataset, RepeatIterDataset does not support
  re-seeding for each epoch. Each epoch will be identical.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      num_epochs: Optional[int] = None,
  ):
    super().__init__(parent)
    if num_epochs is not None and num_epochs <= 0:
      raise ValueError(f"num_epochs must be positive, but got {num_epochs}.")
    self._num_epochs = num_epochs

  def __iter__(self) -> _RepeatDatasetIterator[T]:
    return _RepeatDatasetIterator(self._parent.__iter__(), self._num_epochs)

  def __str__(self) -> str:
    return f"RepeatIterDataset(num_epochs={self._num_epochs})"
