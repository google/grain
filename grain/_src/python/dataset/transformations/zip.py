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

from collections.abc import Sequence
from typing import Any, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats

T = TypeVar("T")


class ZipMapDataset(dataset.MapDataset[T]):
  """Combines MapDatasets of the same length to return a tuple of items.

  At each index, returns a tuple containing the corresponding element from
  each parent dataset. All parent datasets must have the same length;
  otherwise, raises a ``ValueError``.

  Examples:
    Combining corresponding elements from multiple map-style datasets::

      import grain

      # Create two source datasets of equal length.
      inputs_ds = grain.MapDataset.source([10, 20, 30])
      labels_ds = grain.MapDataset.source([40, 50, 60])

      # Combine corresponding elements from both datasets.
      zipped_ds = grain.experimental.ZipMapDataset([inputs_ds, labels_ds])

      print(zipped_ds[0])
      # (10, 40)

      print(zipped_ds[1])
      # (20, 50)
  """

  def __init__(self, parents: Sequence[dataset.MapDataset[T]]):
    """Initializes the ZipMapDataset.

    Args:
      parents: A sequence of MapDatasets to combine. All parent datasets must
        have the same length.

    Raises:
      ValueError: If no parent datasets are provided.
      ValueError: If the parent datasets do not all have the same length.
    """
    super().__init__(parents)
    lengths = [len(p) for p in self._parents]  # pyrefly: ignore[bad-argument-type]
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
    return tuple(p[index] for p in self._parents)  # pyrefly: ignore[bad-index]

  def _getitems(self, indices: Sequence[int]):
    # p._getitems(indices) returns a list of elements of the requested indices.
    # We get a list of lists that we need to zip.
    parent_elements = [
        p._getitems(indices) for p in self.parents  # pylint: disable=protected-access
    ]
    return list(zip(*parent_elements))

  def __str__(self) -> str:
    return f"ZipMapDataset(parents={self._parents}"

  @property
  def _element_spec(self) -> Any:
    return tuple(dataset.get_element_spec(p) for p in self._parents)  # pyrefly: ignore[bad-argument-type]


class ZipIterDataset(dataset.IterDataset[T]):
  """Combines IterDatasets of the same length to return a tuple of items.

  At each iteration, returns a tuple containing the next element from each
  parent dataset. By default (``strict=True``), all parent iterators are
  expected to produce the same number of elements; otherwise, a ``ValueError``
  is raised during iteration. When ``strict=False``, iteration stops when the
  shortest parent iterator is exhausted, matching the behavior of Python's
  built-in ``zip``.

  Example:
    Iterating over corresponding elements from multiple datasets::

      import grain

      # Create two parent pipelines of equal length.
      inputs_ds = grain.MapDataset.source([10, 20, 30]).to_iter_dataset()
      labels_ds = grain.MapDataset.source([40, 50, 60]).to_iter_dataset()

      # Combine corresponding elements from both pipelines.
      zipped_ds = grain.experimental.ZipIterDataset([inputs_ds, labels_ds])
      iterator = iter(zipped_ds)

      print(next(iterator))
      # (10, 40)

      print(next(iterator))
      # (20, 50)
  """

  def __init__(
      self, parents: Sequence[dataset.IterDataset[T]], *, strict: bool = True
  ):
    """Initializes the ZipIterDataset.

    Args:
      parents: A sequence of IterDatasets to combine.
      strict: If ``True`` (default), raises a ``ValueError`` during iteration
        when the parent iterators do not produce the same number of elements. If
        ``False``, iteration stops when the shortest parent iterator is
        exhausted.

    Raises:
      ValueError: If no parent dataset is provided.
    """
    super().__init__(parents)
    if not self._parents:
      raise ValueError("At least one parent must be provided.")
    self._strict = strict

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _ZipDatasetIterator(self._parents, strict=self._strict)  # pyrefly: ignore[bad-argument-type]

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    del sequential_slice
    for parent in self._parents:
      dataset.set_slice(parent, sl)  # pyrefly: ignore[bad-argument-type]

  def __str__(self) -> str:
    return f"ZipIterDataset(parents={self._parents}, strict={self._strict})"

  @property
  def _element_spec(self) -> Any:
    return tuple(dataset.get_element_spec(p) for p in self._parents)  # pyrefly: ignore[bad-argument-type]


def _strict_zip_error(i: int, why: str) -> str:
  plural = " " if i == 1 else "s 1-"
  return f"ZipIterDataset argument {i + 1} is {why} than argument{plural}{i}"


class _ZipDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for ZipIterDataset."""

  def __init__(
      self, parents: Sequence[dataset.IterDataset[T]], *, strict: bool = True
  ):
    super().__init__([p.__iter__() for p in parents])
    self._strict = strict

  @dataset_stats.record_next_duration_if_output
  @dataset_stats.trace_input_pipeline_next(
      stage_category=dataset_stats.IPL_CAT_PREPROCESSING
  )
  def __next__(self) -> tuple[T, ...]:
    with self._stats.record_self_time():
      # Can't use for a `for` loop because we need to raise StopIteration from
      # the inner iterators.
      items = []
      i = 0
      while i < len(self._parents):
        it = self._parents[i]
        try:
          item = next(it)
        except StopIteration as error:
          if self._strict:
            # Check for strict zip violations with similar logic to CPython's
            # zip_traverse
            if i > 0:
              # Previous iterators were not exhausted, so we've already found a
              # violation of strictness.
              raise ValueError(_strict_zip_error(i, "shorter")) from error
            else:
              # Check remaining iterators to make sure they're also exhausted.
              i = 1
              while i < len(self._parents):
                it = self._parents[i]
                try:
                  next(it)
                except StopIteration:
                  pass
                else:
                  raise ValueError(_strict_zip_error(i, "longer")) from error
                i += 1
          raise  # re-raise StopIteration
        items.append(item)
        i += 1
      return self._stats.record_output_spec(tuple(items))

  def get_state(self) -> dict[str, Any]:
    return {"parents": [it.get_state() for it in self._parents]}

  def set_state(self, state):
    for it, s in zip(self._parents, state["parents"]):
      it.set_state(s)

  def get_shard_states(self) -> Sequence[Any]:
    """Returns the shard states for each shard across all parent iterators."""
    parent_shard_states = []
    for parent in self._parents:
      shards_stats = dataset.find_shard_states(parent)
      if not shards_stats:
        raise ValueError(
            f"Parent iterator {parent} does not support elastic resizing."
        )
      parent_shard_states.append(shards_stats)
    if not parent_shard_states:
      return []
    num_shards = len(parent_shard_states[0])
    if not all(len(l) == num_shards for l in parent_shard_states):
      raise ValueError("All parents must have the same number of shards.")
    return list(zip(*parent_shard_states))

  def set_shard_states(self, shard_states: Sequence[Any]) -> None:
    """Restores the shard states for each parent iterator."""
    if not shard_states:
      return
    unzipped_states = list(zip(*shard_states))
    for parent, states in zip(self._parents, unzipped_states):
      dataset.set_shard_states(parent, states)

  def __str__(self) -> str:
    return f"ZipDatasetIterator([{len(self._parents)} parents])"
