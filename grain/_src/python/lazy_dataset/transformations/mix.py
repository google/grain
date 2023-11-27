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

import dataclasses
import functools
import sys
from typing import Any, Sequence, Tuple, TypeVar, Union

from grain._src.core.exceptions import PyGrainInternalError
from grain._src.python.lazy_dataset import lazy_dataset


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


@lazy_dataset.lazy_map_dataset_function("mix")
class MixedLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """LazyDataset for mixtures."""

  def __init__(
      self,
      parents: Sequence[lazy_dataset.LazyMapDataset[T]],
      proportions: Sequence[float | int] | None = None,
  ):
    super().__init__(parents)
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

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    input_index, index = _dataset_and_key_of_next_element(
        index, self._proportions
    )
    return self._parents[input_index][index]


@dataclasses.dataclass
class _MixedLazyDatasetIterator(lazy_dataset.LazyDatasetIterator[T]):
  """Iterator that mixes elements from iterators based on given proportions.

  Note: The current implementation stops sampling elements when any dataset is
  exhausted. This can be extended to allow sampling until all datasets are
  exhausted, either by restarting sampling from the beginning of exhausted
  datasets or deviating from the given proportions.
  """

  def __init__(
      self,
      parents: Sequence[lazy_dataset.LazyDatasetIterator[T]],
      proportions: Sequence[float | int] | None = None,
  ):
    super().__init__()
    self._parents = parents
    self._proportions = tuple(proportions)
    self._index = 0

  def __next__(self):
    input_index, _ = _dataset_and_key_of_next_element(
        self._index, self._proportions
    )
    self._index += 1
    return next(self._parents[input_index])

  def get_state(self):
    return {
        "parents": [parent.get_state() for parent in self._parents],
        "index": self._index,
    }

  def set_state(self, state):
    for parent, parent_state in zip(self._parents, state["parents"]):
      parent.set_state(parent_state)
    self._index = state["index"]

  def __str__(self) -> str:
    return (
        f"MixedLazyDatasetIterator(parents={self._parents},"
        f" proportions={self._proportions})"
    )


@lazy_dataset.lazy_iter_dataset_function("mix")
class MixedLazyIterDataset(lazy_dataset.LazyIterDataset[T]):
  """Mix transformation for LazyIterDatasets."""

  def __init__(
      self,
      parents: Sequence[lazy_dataset.LazyIterDataset],
      proportions: Sequence[float | int] | None = None,
  ):
    super().__init__(parents)
    # Normalize proportions
    if proportions is None:
      proportions = [1] * len(parents)
    elif 0 in proportions:
      raise ValueError("Must specify all non-zero proportions for mixing.")
    else:
      proportions = _float_to_int_proportions(proportions)
    assert len(parents) == len(proportions)
    self._proportions = proportions

  def __iter__(self) -> _MixedLazyDatasetIterator[T]:
    parent_iters = [parent.__iter__() for parent in self._parents]
    return _MixedLazyDatasetIterator(
        parent_iters,
        proportions=self._proportions,
    )

  def __str__(self) -> str:
    return (
        f"MixedLazyIterDataset(parents={self._parents},"
        f" proportions={self._proportions})"
    )


def _float_to_int_proportions(
    values: Sequence[Union[float, int]], scale_min_to: int = 100
) -> Sequence[int]:
  """Scales at values by `scale_min_to/min(proportions)` and cast to int."""
  scale_factor = scale_min_to / min(values)
  return [int(p * scale_factor) for p in values]


@functools.cache
def _counts_per_dataset(k: int, proportions: tuple[int]) -> Sequence[int]:
  """Calculates the counts per dataset at n elements accordings to proportions.

  We are interleaving n infinite datasets into one combined dataset.

  Proportions P is a list of n integers, representing mixing proportions.

  mix(P, k, i) represents the number of examples from component i
  among the first k examples from the mixed sequence. It is given by the
  following formula:

    mix(P, k, 0) = ceiling(k * P[0] / sum(P))
    mix(P, k, i>0) = mix(P[1:], k - mix(P, k, 0), i - 1)

  Element k of the mixed sequence is equal to element m from component i iff:

    mix(P, k + 1, i) == m + 1  AND
    mix(P, k, i) == m

  _counts_per_dataset() computes the "mix" function described above.

  _dataset_and_key_of_next_element() maps from the index in the combined
  dataset to identity of the ID of the source dataset and key in the source
  dataset.

  Args:
    k: Number of elements of the mixed sequence.
    proportions: The mixing proportions for the n dataset.

  Returns:
    Counts of how many elements from each source dataset are used.
  """
  remaining_proportions = sum(proportions)
  result = []
  for p in proportions:
    new_k = (k * (remaining_proportions - p)) // remaining_proportions
    result.append(k - new_k)
    remaining_proportions -= p
    k = new_k
  return result


def _dataset_and_key_of_next_element(
    k: int, proportions: tuple[int]
) -> Tuple[int, int]:
  """Compute the dataset and the key for interleaved datasets at position k.

  We are interleaving n infinite datasets into one combined dataset.

  See the description in _counts_per_dataset() above.

  Args:
    k: Index in the combined dataset.
    proportions: The mixing proportions for the n dataset.

  Returns:
    A tuple with the index of the source dataset and the key in it for the
    element at index `k` of the combined dataset.
  """
  old_counts = _counts_per_dataset(k, proportions)
  new_counts = _counts_per_dataset(k + 1, proportions)
  # For the new dataset the count increased by 1. All other counts should be
  # the same.
  for dataset_index in range(len(proportions)):
    if old_counts[dataset_index] != new_counts[dataset_index]:
      return dataset_index, new_counts[dataset_index] - 1
  raise PyGrainInternalError(
      "PyGrain internal error: please file a bug with the Grain team."
  )
