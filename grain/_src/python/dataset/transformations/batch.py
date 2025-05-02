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
"""Implements batch transformations."""

from __future__ import annotations

from collections.abc import Sequence
import math
import pprint
import sys
from typing import Callable, TypeVar

from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import filter as filter_ds
import numpy as np


T = TypeVar("T")
S = TypeVar("S")


def _make_batch(values: Sequence[T]) -> T:
  """Returns a batch of values with a new batch dimension at the front."""

  if not values:
    raise ValueError("Cannot batch 0 values. Please file a bug.")

  try:
    return tree_lib.map_structure(lambda *xs: np.stack(xs), *values)

  except ValueError as e:
    # NumPy error message doesn't include actual shapes and dtypes. Provide a
    # more helpful error message.
    raise ValueError(
        "Expected all input elements to have the same structure but got:\n"
        f"{pprint.pformat(tree_lib.spec_like(values))}"
    ) from e


class _BatchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that batches elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[S],
      batch_size: int,
      drop_remainder: bool,
      batch_fn: Callable[[Sequence[S]], T],
  ):
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = batch_fn

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    values = []
    for _ in range(self._batch_size):
      try:
        values.append(next(self._parent))
      except StopIteration as e:
        if self._drop_remainder:
          raise e
        break
    if not values:
      raise StopIteration
    with self._stats.record_self_time():
      return self._stats.record_output_spec(self._batch_fn(values))

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return (
        f"BatchDatasetIterator(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


class BatchMapDataset(dataset.MapDataset[T]):
  """Batch transformation for non-sparse MapDatasets."""

  def __init__(
      self,
      parent: dataset.MapDataset[S],
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[S]], T] | None = None,
  ):
    """A MapDataset that batches elements.

    Args:
      parent: The parent MapDataset whose elements are batched.
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new batch dimension.
    """
    super().__init__(parent)
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    to_check = [parent]
    while to_check:
      next_ds = to_check.pop()
      if isinstance(next_ds, filter_ds.FilterMapDataset):
        raise ValueError(
            "`MapDataset.batch` can not follow `MapDataset.filter` "
            "because `filter` can discard elements. Convert `MapDataset` to "
            "`IterDataset` with `to_iter_dataset()` before calling `batch`."
        )
      to_check.extend(next_ds.parents)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = _make_batch if batch_fn is None else batch_fn
    if self._drop_remainder:
      self._length = len(self._parent) // self._batch_size
    else:
      self._length = math.ceil(len(self._parent) / self._batch_size)

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    # Each epoch gets batched separately. If users want to batch across epochs
    # they can repeat() before the batch().
    epoch, index_in_epoch = divmod(index, self._length)
    # Get range within the epoch without going outside the epoch.
    start = index_in_epoch * self._batch_size
    stop = min(len(self._parent), (index_in_epoch + 1) * self._batch_size)
    # Add offset for epoch.
    start += epoch * len(self._parent)
    stop += epoch * len(self._parent)
    values = [self._parent[i] for i in range(start, stop)]
    with self._stats.record_self_time():
      try:
        return self._stats.record_output_spec(self._batch_fn(values))
      except ValueError as e:
        if sys.version_info >= (3, 11):
          e.add_note(
              "\nIf you are trying to batch elements after a sparse "
              "transformation, such as `filter`, you need to first convert the "
              "dataset to `IterDataset` with `to_iter_dataset()` and then "
              "apply `batch`."
          )
        raise e

  def __str__(self) -> str:
    return (
        f"BatchMapDataset(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


class BatchIterDataset(dataset.IterDataset[T]):
  """Batch transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset[S],
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[S]], T] | None = None,
  ):
    """A IterDataset that batches elements.

    Args:
      parent: The parent IterDataset whose elements are batched.
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new batch dimension.
    """
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = _make_batch if batch_fn is None else batch_fn

  def __iter__(self) -> _BatchDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _BatchDatasetIterator(
        parent_iter,
        self._batch_size,
        drop_remainder=self._drop_remainder,
        batch_fn=self._batch_fn,
    )

  def __str__(self) -> str:
    return (
        f"BatchIterDataset(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )
