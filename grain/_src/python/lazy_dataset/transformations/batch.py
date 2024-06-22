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

from collections.abc import Sequence
import math
import pprint
from typing import TypeVar

from grain._src.core import tree
from grain._src.python.lazy_dataset import lazy_dataset
import numpy as np

T = TypeVar("T")


def _make_batch(values: Sequence[T]) -> T:
  """Returns a batch of values with a new batch dimension at the front."""
  if not values:
    raise ValueError("Cannot batch 0 values. Please file a bug.")

  try:
    return tree.map_structure(lambda *xs: np.stack(xs), *values)

  except ValueError as e:
    # NumPy error message doesn't include actual shapes and dtypes. Provide a
    # more helpful error message.
    element_specs = tree.map_structure(
        lambda x: f"{np.asarray(x).dtype}{list(np.asarray(x).shape)}", values
    )
    element_specs = "\n".join(str(x) for x in element_specs)
    raise ValueError(
        "Expected all input elements to have the same structure but got:\n"
        f"{pprint.pformat(element_specs)}"
    ) from e


class _BatchLazyDatasetIterator(lazy_dataset.LazyDatasetIterator[T]):
  """Iterator that batches elements."""

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
      batch_size: int,
      drop_remainder: bool,
  ):
    super().__init__()
    self._parent = parent
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder

  def __next__(self):
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
    return _make_batch(values)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return (
        f"BatchLazyDatasetIterator(parent={self._parent},"
        f" batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


@lazy_dataset.lazy_map_dataset_function("batch")
class BatchLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Batch transformation for non-sparse LazyMapDatasets."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset,
      batch_size: int,
      drop_remainder: bool = False,
  ):
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
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
    return _make_batch(values)

  def __str__(self) -> str:
    return (
        f"BatchMapLazyDataset(parent={self._parent},"
        f" batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


@lazy_dataset.lazy_iter_dataset_function("batch")
class BatchLazyIterDataset(lazy_dataset.LazyIterDataset[T]):
  """Batch transformation for LazyIterDatasets."""

  def __init__(
      self,
      parent: lazy_dataset.LazyIterDataset,
      batch_size: int,
      drop_remainder: bool = False,
  ):
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder

  def __iter__(self) -> _BatchLazyDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _BatchLazyDatasetIterator(
        parent_iter, self._batch_size, drop_remainder=self._drop_remainder
    )

  def __str__(self) -> str:
    return (
        f"BatchIterLazyDataset(parent={self._parent},"
        f" batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )
