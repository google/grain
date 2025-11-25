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
"""Mixing transformation for LazyDataset."""

from __future__ import annotations

import bisect
import collections
from collections.abc import Sequence
import sys
from typing import Any, Mapping, TypeVar

from grain._src.core import exceptions
from grain._src.core import tree_lib
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


class SelectionWithProportionsMap(base.DatasetSelectionMap):
  """A map mixing datasets according to their proportions."""

  def __init__(
      self,
      parents: Sequence[dataset.MapDataset],
      proportions: Sequence[float] | None = None,
  ):
    # Normalize proportions
    if proportions is None:
      proportions = [1] * len(parents)
    elif 0 in proportions:
      raise ValueError("Must specify all non-zero proportions for mixing.")
    else:
      proportions = _float_to_int_proportions(proportions)
    assert len(parents) == len(proportions)
    self._proportions = tuple(proportions)

    # Compute length such that elements of constituent datasets appear at most
    # once.
    weight_sum = sum(proportions)
    lengths = [
        len(parent) / (weight / weight_sum)
        for parent, weight in zip(parents, proportions)
    ]
    self._length = min(sys.maxsize, int(min(lengths)))

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index: int):
    input_index, index = _dataset_and_key_of_next_element(
        index, self._proportions
    )
    return input_index, index

  def __repr__(self):
    return (
        f"SelectionWithProportionsMap(proportions={self._proportions}, "
        f"length={self._length})"
    )


class MixedMapDataset(dataset.MapDataset[T]):
  """LazyDataset for mixtures."""

  def __init__(
      self,
      parents: Sequence[dataset.MapDataset[T]],
      proportions: Sequence[float] | None = None,
      selection_map: base.DatasetSelectionMap | None = None,
  ):
    """Initializes the mixed dataset.

    Args:
      parents: Component datasets to draw from.
      proportions: Proportions from which to draw from each parent dataset.
        Defaults to uniform weight if selection_map is not given.
      selection_map: Mapping from global index to paraent dataset and index
        within parent dataset.
    """
    super().__init__(parents)
    # Cannot set both proportions and selection_map
    if proportions is not None and selection_map is not None:
      raise ValueError("Cannot set both proportions and selection_map.")

    if selection_map is None:
      selection_map = SelectionWithProportionsMap(parents, proportions)

    self._selection_map = selection_map

    self._length = len(self._selection_map)

  def __len__(self) -> int:
    return self._length

  def __str__(self):
    return f"MixedMapDataset[{len(self._parents)} parents]"

  def __repr__(self):
    # Note that we use parent lengths instead of parent representations because
    # the component datasets will likely not have `__repr__` implemented
    # resulting in unstable output.
    return (
        f"MixedMapDataset(parent_lengths={[len(p) for p in self._parents]}, "
        f"selection_map={repr(self._selection_map)})"
    )

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      dataset_index, index_in_dataset = self._selection_map[index]
    try:
      return self._stats.record_output_spec(
          self._parents[dataset_index][index_in_dataset]
      )
    except Exception as e:
      if sys.version_info >= (3, 11):
        e.add_note(
            f"Exception caught while processing dataset @ {dataset_index=},"
        )
      raise e

  def _getitems(self, indices: Sequence[int]) -> Sequence[T | None]:
    """Returns a sequence of elements corresponding to the given indices."""

    with self._stats.record_self_time(num_elements=len(indices)):
      parent_and_key_pairs = (self._selection_map[i] for i in indices)

      # Group indices by parent.
      # keys_by_parent will store {parent_idx: [key1, key2, ...]}
      # where key is the index in the parent dataset.
      keys_by_parent = collections.defaultdict(list)
      # original_indices_by_parent will store {parent_idx: [original_idx1, ...]}
      # where original_idx is the position in the input `indices` list.
      original_indices_by_parent = collections.defaultdict(list)

      for i, (parent_idx, key) in enumerate(parent_and_key_pairs):
        keys_by_parent[parent_idx].append(key)
        original_indices_by_parent[parent_idx].append(i)

    mixed_elements = [None] * len(indices)
    for parent_idx, keys in keys_by_parent.items():
      # Call _getitems for each parent.
      parent_elements = self.parents[parent_idx]._getitems(keys)  # pylint: disable=protected-access
      original_indices = original_indices_by_parent[parent_idx]
      # Assign elements to their proper index in the final mixed dataset.
      for i, element in enumerate(parent_elements):
        mixed_elements[original_indices[i]] = element

    return self._stats.record_output_spec_for_batch(mixed_elements)


class _MixedDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that mixes elements from iterators based on given proportions.

  Note: The current implementation stops sampling elements when any dataset is
  exhausted. This can be extended to allow sampling until all datasets are
  exhausted, either by restarting sampling from the beginning of exhausted
  datasets or deviating from the given proportions.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parents: (
          Sequence[dataset.DatasetIterator[T]]
          | Mapping[str, dataset.DatasetIterator[T]]
      ),
      proportions: Sequence[int],
  ):
    flat_parents = tree_lib.flatten(parents)
    super().__init__(flat_parents)
    self._parents_structure = parents
    self._proportions = tuple(proportions)
    self._index = 0
    self._stop = False

  @stats.record_next_duration_if_output
  def __next__(self):
    if self._stop:
      # Although there may be elements available in some parent datasets, do not
      # sample once stop signal is turned on.
      raise StopIteration
    with self._stats.record_self_time():
      input_index, _ = _dataset_and_key_of_next_element(
          self._index, self._proportions
      )
    self._index += 1
    try:
      elem = next(self._parents[input_index])
    except Exception as e:
      # Turn on stop signal as soon as the end of any dataset is reached.
      self._stop = True
      if sys.version_info >= (3, 11):
        e.add_note(
            f"Exception caught while processing dataset @ {input_index=}"
        )
      raise e
    return self._stats.record_output_spec(elem)

  def get_state(self):
    return {
        "parents": tree_lib.map_structure(
            lambda p: p.get_state(), self._parents_structure
        ),
        "index": self._index,
        "stop": self._stop,
    }

  def set_state(self, state):
    parents_state = state["parents"]
    if isinstance(self._parents_structure, Sequence):
      for parent, parent_state in zip(self._parents, parents_state):
        parent.set_state(parent_state)
    else:
      for key, parent in self._parents_structure.items():
        if (parent_state := parents_state.get(key)) is not None:
          parent.set_state(parent_state)

    self._index = state["index"]
    self._stop = state["stop"]

  def __str__(self) -> str:
    return (
        f"MixedDatasetIterator([{len(self._parents)} parents],"
        f" proportions={self._proportions})"
    )


class MixedIterDataset(dataset.IterDataset[T]):
  """Mix transformation for IterDatasets."""

  def __init__(
      self,
      parents: (
          Sequence[dataset.IterDataset] | Mapping[str, dataset.IterDataset]
      ),
      proportions: Sequence[float] | Mapping[str, float] | None = None,
  ):
    flat_parents = tree_lib.flatten(parents)
    super().__init__(flat_parents)
    self._parents_structure = parents
    # Normalize proportions
    if proportions is None:
      proportions = [1] * len(flat_parents)
    else:
      proportions = tree_lib.flatten(proportions)

    if 0 in proportions:
      raise ValueError("Must specify all non-zero proportions for mixing.")
    else:
      proportions = _float_to_int_proportions(proportions)
    assert len(flat_parents) == len(proportions)
    self._proportions = proportions

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _MixedDatasetIterator(
        tree_lib.map_structure(lambda p: p.__iter__(), self._parents_structure),
        proportions=self._proportions,
    )

  def __str__(self) -> str:
    return (
        f"MixedIterDataset([{len(self._parents)} parents],"
        f" proportions={self._proportions})"
    )


def _float_to_int_proportions(
    values: Sequence[float], scale_min_to: int = 100
) -> Sequence[int]:
  """Scales at values by `scale_min_to/min(proportions)` and cast to int."""
  scale_factor = scale_min_to / min(values)
  return [int(p * scale_factor) for p in values]


def _dataset_and_key_of_next_element(
    k: int, proportions: tuple[int, ...]
) -> tuple[int, int]:
  """Compute the dataset and the key for interleaved datasets at position k.

  We are interleaving n infinite datasets into one combined dataset.

  We determine which dataset provides the (k+1)-th element
  in the mixed sequence and what the index within that dataset is.
  We find the dataset at which the frequency count increases.

  Args:
    k: Index in the combined dataset.
    proportions: The mixing proportions for the n dataset.

  Returns:
    A tuple with the index of the source dataset and the key in it for the
    element at index `k` of the combined dataset.
  """

  # TODO: if we're happy to be approximate, this could be faster
  remaining = sum(proportions)
  curr_k, curr_k1 = k, k + 1
  for i, p in enumerate(proportions):
    next_k = (curr_k * (remaining - p)) // remaining
    next_k1 = (curr_k1 * (remaining - p)) // remaining
    count_k_plus_1 = curr_k1 - next_k1
    if (curr_k - next_k) != count_k_plus_1:
      return i, count_k_plus_1 - 1
    remaining -= p
    curr_k, curr_k1 = next_k, next_k1

  raise exceptions.PyGrainInternalError(
      "PyGrain internal error: please file a bug with the Grain team."
  )


class _ConcatSelectionMap(base.DatasetSelectionMap):
  """Concatenated datasets selection map.

  Selection map that concatenates the elements from a sequence of finite parent
  datasets. Elements from a dataset with index i will appear after the elements
  of all previous datasets with indices 0, 1, ..., i-1 and before the elements
  of datasets with indices i+1, i+2, ...
  """

  def __init__(self, parents: Sequence[dataset.MapDataset]):
    dataset_sizes = [len(parent) for parent in parents]
    for i, dataset_size in enumerate(dataset_sizes):
      if dataset_size >= sys.maxsize:
        raise ValueError(
            f"Cannot concatenate infinite datasets. {parents[i]} is infinite."
        )
    cumulative_sizes = [0] * (len(parents) + 1)
    for i in range(len(parents)):
      cumulative_sizes[i + 1] = cumulative_sizes[i] + dataset_sizes[i]
    self._cumulative_dataset_sizes = cumulative_sizes
    self._dataset_sizes = dataset_sizes

  def __len__(self) -> int:
    return self._cumulative_dataset_sizes[-1]

  def __getitem__(self, index: int) -> tuple[int, int]:
    epoch, index_in_epoch = divmod(index, len(self))
    dataset_index = (
        bisect.bisect_right(self._cumulative_dataset_sizes, index_in_epoch) - 1
    )
    epochs_offset = epoch * self._dataset_sizes[dataset_index]
    index_in_dataset_in_epoch = (
        index_in_epoch - self._cumulative_dataset_sizes[dataset_index]
    )
    return dataset_index, index_in_dataset_in_epoch + epochs_offset


class ConcatenateMapDataset(MixedMapDataset[T]):
  """MapDataset for concatenating the elements from a sequence of datasets."""

  def __init__(
      self,
      parents: Sequence[dataset.MapDataset[T]],
  ):
    """Initializes the concatenated dataset.

    Args:
      parents: Component datasets to draw from. We will draw elements in order
        of their appearance in `parents`.
    """
    super().__init__(parents, selection_map=_ConcatSelectionMap(parents))
