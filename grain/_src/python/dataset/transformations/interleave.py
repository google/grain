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
"""Implements dataset interleaving."""

from collections.abc import Sequence
from typing import TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import prefetch

T = TypeVar("T")


def _add_prefetch_and_make_iterator(
    ds: dataset.IterDataset[T] | dataset.MapDataset[T],
) -> dataset.DatasetIterator[T]:
  if isinstance(ds, dataset.MapDataset):
    # Prefetch is automatically added in `MapDataset.__iter__`.
    return ds.__iter__()
  return prefetch.ThreadPrefetchIterDataset(
      ds, prefetch_buffer_size=1
  ).__iter__()


class _InterleaveDatasetIterator(dataset.DatasetIterator[T]):
  """Iterates over the interleaved datasets."""

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      cycle_length: int,
  ):
    # `datasets` is allowed to be a lazily evaluated `MapDataset`. We avoid
    # passing it as `parents` to not trigger evaluation early.
    super().__init__()
    self._datasets = datasets
    self._cycle_length: int = min(cycle_length, len(datasets))
    self._next_index_in_cycle: int = 0
    self._next_index_in_datasets: int = 0
    self._iterators_in_use_indices: list[int] = list(range(self._cycle_length))
    self._iterators_in_use: list[dataset.DatasetIterator[T] | None] = [
        None
    ] * self._cycle_length

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    while True:
      if iterator_to_use := self._iterators_in_use[self._next_index_in_cycle]:
        try:
          result = iterator_to_use.__next__()
          self._next_index_in_cycle = (
              self._next_index_in_cycle + 1
          ) % self._cycle_length
          return result
        except StopIteration:
          self._iterators_in_use[self._next_index_in_cycle] = None
          continue
      if self._next_index_in_datasets < len(self._datasets):
        self._iterators_in_use[self._next_index_in_cycle] = (
            _add_prefetch_and_make_iterator(
                self._datasets[self._next_index_in_datasets]
            )
        )
        self._iterators_in_use_indices[self._next_index_in_cycle] = (
            self._next_index_in_datasets
        )
        self._next_index_in_datasets += 1
      elif not any(self._iterators_in_use):
        raise StopIteration
      else:
        self._next_index_in_cycle = (
            self._next_index_in_cycle + 1
        ) % self._cycle_length

  def get_state(self):
    return {
        "next_index_in_cycle": self._next_index_in_cycle,
        "next_index_in_datasets": self._next_index_in_datasets,
        "iterators_in_use_indices": self._iterators_in_use_indices.copy(),
        "iterators_in_use_states": [
            (None if it is None else it.get_state())
            for it in self._iterators_in_use
        ],
    }

  def set_state(self, state):
    self._next_index_in_cycle = state["next_index_in_cycle"]
    self._next_index_in_datasets = state["next_index_in_datasets"]
    if not self._next_index_in_datasets and not self._next_index_in_cycle:
      return
    self._iterators_in_use_indices = state["iterators_in_use_indices"]
    for index_in_cycle, (index_in_datasets, it_state) in enumerate(
        zip(self._iterators_in_use_indices, state["iterators_in_use_states"])
    ):
      if it_state is None:
        self._iterators_in_use[index_in_cycle] = None
      else:
        iterator = self._datasets[index_in_datasets].__iter__()
        iterator.set_state(it_state)
        self._iterators_in_use[index_in_cycle] = iterator

  def __str__(self) -> str:
    return (
        f"InterleaveDatasetIterator([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )


class InterleaveIterDataset(dataset.IterDataset[T]):
  """Interleaves the given sequence of datasets.

  The sequence can be a `MapDataset`.

  Creates at most `cycle_length` iterators at a time that are processed
  concurrently and interleives their elements. If `cycle_length` is larger than
  the number of datasets, then the behavior is similar to mixing the datasets
  with equal proportions. If `cycle_length` is 1, the datasets are chained.

  Can be used with `mp_prefetch` to parallelize reading from sources that do not
  support random access and are implemented as `IterDataset`::

    def make_source(filename: str) -> grain.IterDataset:
      ...

    ds = grain.MapDataset.source(filenames).shuffle(seed=42).map(make_source)
    ds = grain.experimental.InterleaveIterDataset(ds, cycle_length=4)
    ds = ...
    ds = ds.mp_prefetch(ds, 2)
    for element in ds:
      ...
  """

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      *,
      cycle_length: int,
  ):
    super().__init__()
    self._datasets = datasets
    self._cycle_length = cycle_length

  def __iter__(self) -> _InterleaveDatasetIterator[T]:
    return _InterleaveDatasetIterator(
        self._datasets,
        cycle_length=self._cycle_length,
    )

  def set_slice(self, sl: slice):
    self._datasets = self._datasets[sl]

  def __str__(self) -> str:
    return (
        f"InterleaveIterDataset([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )
