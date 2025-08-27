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
"""Implements rebatch transformation."""

from __future__ import annotations

from typing import Any

from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
import numpy as np


class RebatchIterDataset(dataset.IterDataset):
  """Rebatches the input PyTree elements."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      batch_size: int,
      drop_remainder: bool = False,
  ):
    """An IterDataset that rebatches elements.

    Args:
      parent: The parent IterDataset whose elements are to be rebatched.
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
    """
    super().__init__(parent)
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder

  def __iter__(self) -> dataset.DatasetIterator:
    parent_iter = self._parent.__iter__()
    return _RebatchDatasetIterator(
        parent_iter,
        self._batch_size,
        drop_remainder=self._drop_remainder,
    )

  def __str__(self) -> str:
    return (
        f"RebatchIterDataset(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


class _RebatchDatasetIterator(dataset.DatasetIterator):
  """Iterator that rebatches elements to a new batch size."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      batch_size: int,
      drop_remainder: bool = False,
  ):
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._current_batch: list[Any] | None = None
    self._batch_idx = 0
    self._treedef = None
    self._last_parent_state = self._parent.get_state()

  def get_state(self) -> dict[str, Any]:
    return {
        "parent": self._last_parent_state,
        "batch_idx": self._batch_idx,
    }

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(state["parent"])
    self._last_parent_state = self._parent.get_state()
    self._batch_idx = state["batch_idx"]
    self._current_batch = None

  def _flatten(self, pytree_element):
    """Flattens the pytree element, if first time, caches the tree def."""
    if self._treedef is None:
      self._treedef = pytree_element
    return tree_lib.flatten(pytree_element)

  @stats.record_next_duration_if_output
  def __next__(self):
    timer = stats.Timer()

    batch_dim_needed = self._batch_size
    to_concatenate: list[list[Any]] = []

    while batch_dim_needed > 0:

      # Gets the next batch when there is no batch or the `_batch_idx` is
      # already past the current batch's batch dimension
      if self._current_batch is None or self._batch_idx >= len(
          self._current_batch[0]
      ):
        # if the iterator is just created or state was set, do not reset the
        # batch index.
        is_first_batch = self._current_batch is None

        # When out of elements, emit the last batch if `to_concatenate` contains
        # data and `drop_remainder` is False.
        try:
          self._last_parent_state = self._parent.get_state()
          element = next(self._parent)
          with timer:
            self._current_batch = self._flatten(element)
        except StopIteration as exc:
          if self._drop_remainder:
            raise StopIteration from exc
          break
        self._batch_idx = self._batch_idx if is_first_batch else 0

      with timer:
        if len(self._current_batch[0]) <= 0:
          raise ValueError(
              "Rebatching with starting batch size of 0 is not supported."
          )

        if not to_concatenate:
          for _ in range(len(self._current_batch)):
            to_concatenate.append(list())

        # Checks if the current batch contains enough data to slice
        if self._batch_idx + batch_dim_needed <= len(self._current_batch[0]):
          for i in range(len(self._current_batch)):
            to_concatenate[i].append(
                self._current_batch[i][
                    self._batch_idx : self._batch_idx + batch_dim_needed
                ]
            )

          self._batch_idx += batch_dim_needed
          batch_dim_needed -= batch_dim_needed
        else:
          # The current batch does not have enough data
          batch_dim = len(self._current_batch[0]) - self._batch_idx

          for i in range(len(self._current_batch)):
            to_concatenate[i].append(self._current_batch[i][self._batch_idx :])

          self._batch_idx += batch_dim
          batch_dim_needed -= batch_dim

    with self._stats.record_self_time(offset_ns=timer.value()):
      if not to_concatenate:
        raise StopIteration
      result_leaves: list[np.ndarray] = []
      for i in range(len(to_concatenate)):
        if len(to_concatenate[i]) == 1:
          # Only one element, nothing to concatenate
          result_leaves.append(to_concatenate[i][0])
        else:
          result_leaves.append(np.concatenate(to_concatenate[i], axis=0))

      return self._stats.record_output_spec(
          tree_lib.unflatten_as(self._treedef, result_leaves)
      )
