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
import functools
from typing import Any, TypeVar
import weakref

from grain._src.python import options as grain_options
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import prefetch


T = TypeVar("T")


class _InterleaveDatasetIterator(dataset.DatasetIterator[T]):
  """Iterates over the interleaved datasets."""

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      cycle_length: int,
      num_make_iter_threads: int = 1,
      make_iter_buffer_size: int = 1,
      iter_buffer_size: int = 1,
  ):
    # `datasets` is allowed to be a lazily evaluated `MapDataset`. We avoid
    # passing it as `parents` to not trigger evaluation early.
    super().__init__()
    self._datasets = datasets
    self._num_make_iter_threads = num_make_iter_threads
    self._make_iter_buffer_size = make_iter_buffer_size
    self._iter_buffer_size = iter_buffer_size
    self._prefetch_ds_iter = (
        dataset.MapDataset.source(datasets)
        .map(
            functools.partial(
                _add_prefetch_and_make_iterator,
                # We use weakref to avoid a circular reference. The
                # _InterleaveDatasetIterator holds a reference to the
                # prefetch iterator in `self._prefetch_ds_iter`.
                # The call to `_add_prefetch_and_make_iterator` (and the
                # partial object) would hold a reference to the
                # _InterleaveDatasetIterator. This would prolong its lifetime
                # leading to increased resource usage.
                interleave_iterator=weakref.ref(self),
                start_prefetch=True,
            )
        )
        .to_iter_dataset(
            grain_options.ReadOptions(
                num_threads=self._num_make_iter_threads,
                prefetch_buffer_size=self._make_iter_buffer_size,
            )
        )
        .__iter__()
    )
    self._cycle_length: int = min(cycle_length, len(datasets))
    self._next_index_in_cycle: int = 0
    self._next_index_in_datasets: int = 0
    self._iterators_in_use_indices: list[int] = list(range(self._cycle_length))
    self._iterators_in_use: list[dataset.DatasetIterator[T] | None] = [
        None
    ] * self._cycle_length
    self._exhausted_iterator_state: list[dict[str, Any] | None] = [
        None
    ] * self._cycle_length
    self._started = False

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self) -> T:
    self._assert_not_closed()
    self._started = True
    while True:
      if iterator_to_use := self._iterators_in_use[self._next_index_in_cycle]:
        try:
          result = iterator_to_use.__next__()
          self._next_index_in_cycle = (
              self._next_index_in_cycle + 1
          ) % self._cycle_length
          return result
        except StopIteration:
          self._exhausted_iterator_state[self._next_index_in_cycle] = (
              iterator_to_use.get_state()
          )
          self._iterators_in_use[self._next_index_in_cycle] = None
          self._next_index_in_cycle = (
              self._next_index_in_cycle + 1
          ) % self._cycle_length
          continue
      if self._next_index_in_datasets < len(self._datasets):
        self._iterators_in_use[self._next_index_in_cycle] = next(
            self._prefetch_ds_iter
        )
        self._exhausted_iterator_state[self._next_index_in_cycle] = None
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
    iterators_in_use_states = [None] * self._cycle_length
    for i in range(self._cycle_length):
      it = self._iterators_in_use[i]
      if it is not None:
        iterators_in_use_states[i] = it.get_state()
      elif self._exhausted_iterator_state[i] is not None:
        iterators_in_use_states[i] = self._exhausted_iterator_state[i]
      elif self._next_index_in_datasets >= len(self._datasets):
        break
      else:
        if self._started:
          it = next(self._prefetch_ds_iter)
        else:
          it = _add_prefetch_and_make_iterator(
              self._datasets[self._next_index_in_datasets],
              interleave_iterator=weakref.ref(self),
              start_prefetch=self._started,
          )
        self._iterators_in_use[i] = it
        iterators_in_use_states[i] = it.get_state()
        self._iterators_in_use_indices[i] = self._next_index_in_datasets
        self._next_index_in_datasets += 1
    if not self._started:
      self._prefetch_ds_iter.set_state(
          {"next_index": self._next_index_in_datasets}
      )
    # Use int instead of bool as it is friendly for Pathways remote python.
    exhausted = [
        int(self._exhausted_iterator_state[i] is not None)
        for i in range(self._cycle_length)
    ]
    return {
        "next_index_in_cycle": self._next_index_in_cycle,
        "next_index_in_datasets": self._next_index_in_datasets,
        "iterators_in_use_indices": self._iterators_in_use_indices.copy(),
        "iterators_in_use_states": iterators_in_use_states,
        "exhausted": exhausted,
    }

  def set_state(self, state):
    self._prefetch_ds_iter.set_state(
        {"next_index": state["next_index_in_datasets"]}
    )
    self._next_index_in_cycle = state["next_index_in_cycle"]
    self._next_index_in_datasets = state["next_index_in_datasets"]
    self._iterators_in_use_indices = state["iterators_in_use_indices"]
    exhausted = state["exhausted"]
    for index_in_cycle, (index_in_datasets, it_state) in enumerate(
        zip(self._iterators_in_use_indices, state["iterators_in_use_states"])
    ):
      if it_state is None:
        self._iterators_in_use[index_in_cycle] = None
      elif exhausted[index_in_cycle] == 0:
        iterator = _add_prefetch_and_make_iterator(
            self._datasets[index_in_datasets],
            interleave_iterator=weakref.ref(self),
            start_prefetch=False,
        )
        iterator.set_state(it_state)
        self._iterators_in_use[index_in_cycle] = iterator
      else:
        self._exhausted_iterator_state[index_in_cycle] = it_state
        self._iterators_in_use[index_in_cycle] = None

  def close(self) -> None:
    """Closes the iterator and shuts down the iterator prefetching."""
    if self._closed:
      return
    self._closed = True
    self._prefetch_ds_iter.close()
    for iterator in self._iterators_in_use:
      if iterator is not None:
        iterator.close()

  def __str__(self) -> str:
    return (
        f"InterleaveDatasetIterator([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )


def _add_prefetch_and_make_iterator(
    ds: dataset.IterDataset[T] | dataset.MapDataset[T],
    interleave_iterator: weakref.ref[_InterleaveDatasetIterator[T]],
    start_prefetch: bool,
) -> dataset.DatasetIterator[T]:
  """Adds prefetching to an IterDataset and returns an iterator.

  If the input is a MapDataset, prefetching is handled by `MapDataset.__iter__`.
  If the input is an IterDataset, a `ThreadPrefetchIterDataset` is used to
  add prefetching.

  Args:
    ds: The dataset to create an iterator from.
    interleave_iterator: The `InterleaveDatasetIterator` instance.
    start_prefetch: Whether to start the prefetching on iterator creation.

  Returns:
    A `dataset.DatasetIterator` for the given dataset, with prefetching
    enabled if applicable.

  Raises:
    RuntimeError: If the interleave_iterator has been garbage collected.
  """
  interleave_iterator_obj = interleave_iterator()
  if interleave_iterator_obj is None:
    raise RuntimeError("InterleaveDatasetIterator has been garbage collected.")
  if isinstance(ds, dataset.MapDataset):
    # Prefetch is automatically added in `MapDataset.__iter__`.
    return ds.__iter__()
  iterator = prefetch.ThreadPrefetchIterDataset(
      ds, prefetch_buffer_size=interleave_iterator_obj._iter_buffer_size  # pylint: disable=protected-access
  ).__iter__()
  # Propagate options applied after InterleaveIterDataset to the iterators that
  # are being interleaved.
  iterator._ctx.dataset_options = interleave_iterator_obj._ctx.dataset_options.merge(iterator._ctx.dataset_options)  # pylint: disable=protected-access
  if start_prefetch:
    iterator.start_prefetch()
  return iterator


class InterleaveIterDataset(dataset.IterDataset[T]):
  """Interleaves the given sequence of datasets.

  The sequence can be a `MapDataset`.

  Concurrently processes at most `cycle_length` iterators and interleaves their
  elements. If `cycle_length` is larger than the number of datasets, then the
  behavior is similar to mixing the datasets with equal proportions. If
  `cycle_length` is 1, the datasets are chained.

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

  Element spec inference assumes that input datasets have the same element spec.
  """

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      *,
      cycle_length: int,
      num_make_iter_threads: int = 1,
      make_iter_buffer_size: int = 1,
      iter_buffer_size: int = 1,
  ):
    """Initializes the InterleaveIterDataset.

    Args:
      datasets: A sequence of `IterDataset` or `MapDataset` objects, or a
        `MapDataset` of datasets to be interleaved.
      cycle_length: The maximum number of input datasets from which elements
        will be processed concurrently. If `cycle_length` is greater than the
        total number of datasets, all available datasets will be interleaved. If
        `cycle_length` is 1, the datasets will be processed sequentially.
      num_make_iter_threads: Optional. The number of threads to use for
        asynchronously creating new iterators and starting prefetching elements
        (for each iterator) from the underlying datasets. Default value is 1,
        with this we'll create one background thread to asynchronously create
        iterators.
      make_iter_buffer_size: Optional. The number of iterators to create and
        keep ready in advance in each preparation thread. This helps in reducing
        latency by ensuring iterators are available when needed. Default value
        is 1, with this we'll always keep the next iterator ready in advance.
      iter_buffer_size: Optional. The number of elements to prefetch from each
        iterator. Default value is 1.
    """
    super().__init__()
    self._datasets = datasets
    self._cycle_length = cycle_length
    self._num_make_iter_threads = num_make_iter_threads
    self._make_iter_buffer_size = make_iter_buffer_size
    self._iter_buffer_size = iter_buffer_size

  def __iter__(self) -> _InterleaveDatasetIterator[T]:
    return _InterleaveDatasetIterator(
        self._datasets,
        cycle_length=self._cycle_length,
        num_make_iter_threads=self._num_make_iter_threads,
        make_iter_buffer_size=self._make_iter_buffer_size,
        iter_buffer_size=self._iter_buffer_size,
    )

  def set_slice(self, sl: slice, sequential_slice: bool = False):
    del sequential_slice
    self._datasets = self._datasets[sl]

  def __str__(self) -> str:
    return (
        f"InterleaveIterDataset([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )

  @property
  def _element_spec(self) -> Any:
    # Assumes that interleaved datasets have the same element spec.
    return dataset.get_element_spec(self._datasets[0])
