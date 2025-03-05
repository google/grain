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
"""Implements global shuffle transformation."""

from __future__ import annotations

import copy
from typing import TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.experimental.index_shuffle.python import index_shuffle_module as index_shuffle


T = TypeVar("T")


class ShuffleMapDataset(dataset.MapDataset[T]):
  """Shuffles the parent dataset."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      *,
      seed: int | None = None,
  ):
    super().__init__(parent)
    seed = self._default_seed if seed is None else seed
    if seed is None:
      raise ValueError(
          "`shuffle` requires a seed. Please provide it with `ds.seed(seed)`"
      )
    if seed < 0 or seed >= 2**32:
      raise ValueError(
          f"Seed must be an integer between 0 and 2**32-1 (got {seed=})."
      )
    self._seed = int(seed)

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return "ShuffleMapDataset"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      length = len(self._parent)
      epoch, index_in_epoch = divmod(index, length)
      # Note:
      #   - index_shuffle expects 32-bit integers
      #   - we use different seeds for each epoch to ensure that the shuffle is
      #     different for each epoch
      per_epoch_seed = (self._seed + epoch) % 2**32
      shuffled_index_in_epoch = index_shuffle.index_shuffle(
          index_in_epoch, max_index=length - 1, seed=per_epoch_seed, rounds=4
      )
      shuffled_index = shuffled_index_in_epoch + epoch * length
    return self._parent[shuffled_index]


class WindowShuffleMapDataset(dataset.MapDataset[T]):
  """Shuffles the parent dataset within a given window.

  Shuffles the retrieval index within a range, given by window_size. Each unique
  index corresponds to exactly one shuffled index (i.e. there is a one-to-one
  mapping and hence a guarantee that no shuffled indices are repeated within a
  given window).
  """

  def __init__(
      self, parent: dataset.MapDataset, *, window_size: int, seed: int
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return "WindowShuffleMapDataset"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      window_index, index_in_window = divmod(index, self._window_size)
      seed = self._seed + window_index
      index_in_window = index_shuffle.index_shuffle(
          index_in_window,
          max_index=self._window_size - 1,
          seed=seed,
          rounds=4,
      )
      index = index_in_window + window_index * self._window_size
    return self._stats.record_output_spec(self._parent[index])


class WindowShuffleIterDataset(dataset.IterDataset[T]):
  """Shuffles the parent dataset within a given window.

  Fetches `window_size` elements from the parent iterator and returns them in
  shuffled order. Each window is shuffled with different seed derived from the
  input seed.
  """

  def __init__(
      self, parent: dataset.IterDataset, *, window_size: int, seed: int
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._seed = seed

  def __iter__(self) -> _WindowShuffleDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _WindowShuffleDatasetIterator(
        parent_iter, window_size=self._window_size, seed=self._seed
    )

  def __str__(self) -> str:
    return "WindowShuffleIterDataset"


class _WindowShuffleDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that within a window shuffles elements."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      window_size: int,
      seed: int,
  ):
    super().__init__(parent)
    self._window_size = window_size
    self._global_seed = seed
    self._window_index: int = 0
    self._pos_in_window = 0
    self._window: list[T] = []
    self._parent_window_start_iter_state = self._parent.get_state()
    self._init = True
    self._parent_exhausted = False

  def _maybe_update_window_index(self):
    # Ugly workaround to allow for the initialization upon calling next and not
    # creating the new iterator.
    if self._init:
      self._init = False
    else:
      self._window_index += 1

  def _reshuffle_list(self, seed: int, window: list[T]):
    window_len = len(window)
    shuffled = []
    for pos in range(window_len):
      shuffled_index = index_shuffle.index_shuffle(
          pos, max_index=window_len - 1, seed=seed, rounds=4
      )
      shuffled.append(window[shuffled_index])
    return shuffled

  def _fill_and_shuffle_window(self):
    # Window should be empty at this point.
    self._window = []
    try:
      for _ in range(self._window_size):
        self._window.append(next(self._parent))
    except StopIteration:
      # End of the parent iterator, nothing else to process.
      self._parent_exhausted = True
    self._window = self._reshuffle_list(
        seed=self._global_seed + self._window_index, window=self._window
    )

  @stats.record_next_duration_if_output
  def __next__(self):
    # Window is empty, fill up the next window.
    if not self._window:
      if self._parent_exhausted:
        raise StopIteration
      # Checkpoints require reshuffling the window regardless the progress
      # within it. Store the parent window start.
      self._parent_window_start_iter_state = self._parent.get_state()
      self._maybe_update_window_index()
      self._fill_and_shuffle_window()
      self._pos_in_window = 0
    # If the window is empty after reshuffling means no elements are left.
    if not self._window:
      raise StopIteration
    self._pos_in_window += 1
    return self._window.pop()

  def get_state(self):
    return dict(
        parent_window_start_state=copy.deepcopy(
            self._parent_window_start_iter_state
        ),
        window_index=self._window_index,
        pos_in_window=self._pos_in_window,
        parent_exhausted=self._parent_exhausted,
    )

  def set_state(self, state):
    self._parent_window_start_iter_state = state["parent_window_start_state"]
    self._parent.set_state(self._parent_window_start_iter_state)
    self._window_index = state["window_index"]
    self._pos_in_window = state["pos_in_window"]
    self._parent_exhausted = state["parent_exhausted"]
    self._fill_and_shuffle_window()
    # Removed previously processed elements from the window.
    for _ in range(min(self._pos_in_window, len(self._window))):
      self._window.pop()

  def __str__(self) -> str:
    return "WindowShuffleDatasetIterator"
