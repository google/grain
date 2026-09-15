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

from __future__ import annotations

import collections
from collections.abc import Sequence
import copy
import functools
from typing import Any, TypeVar, cast
import weakref

from absl import logging
from concurrent import futures
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import autotune
from grain._src.python.dataset.transformations import prefetch

T = TypeVar("T")


class InterleaveDatasetIterator(dataset.DatasetIterator[T]):
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
    self._prefetch_ds_iter = (  # pyrefly: ignore[invalid-type-var]
        dataset.MapDataset.source(datasets)
        .map(
            functools.partial(
                _add_prefetch_and_make_iterator,
                # We use weakref to avoid a circular reference. The
                # InterleaveDatasetIterator holds a reference to the
                # prefetch iterator in `self._prefetch_ds_iter`.
                # The call to `_add_prefetch_and_make_iterator` (and the
                # partial object) would hold a reference to the
                # InterleaveDatasetIterator. This would prolong its lifetime
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
    self._parent_stats: list[stats.Stats | None] = [None] * self._cycle_length
    self._keep_iterators_after_stop_iteration = False
    self._exhausted_iterators: list[
        tuple[int, dataset.DatasetIterator[T]] | None
    ] = [None] * self._cycle_length
    # Future states used for elastic iterators
    self._future_states: dict[int, Any] = {}
    self._cached_start_states: dict[int, Any] = {}

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self) -> T:
    self._assert_not_closed()
    self._started = True
    timer = stats.Timer()
    _ = self._stats  # eagerly initialize stats
    while True:
      if iterator_to_use := self._iterators_in_use[self._next_index_in_cycle]:
        try:
          result = iterator_to_use.__next__()
          # Get the stats object of the iterator that is being used.
          # pylint: disable=protected-access
          with self._stats.record_self_time(offset_ns=timer.value()):
            if (
                self._parent_stats[self._next_index_in_cycle] is None
                or self._parent_stats[self._next_index_in_cycle]
                is not iterator_to_use._stats
            ):
              self._parent_stats[self._next_index_in_cycle] = (
                  iterator_to_use._stats
              )
              self._stats._parents = [
                  p for p in self._parent_stats if p is not None
              ]
            # pylint: enable=protected-access
            self._next_index_in_cycle = (
                self._next_index_in_cycle + 1
            ) % self._cycle_length
            result = self._stats.record_bytes_produced(result)
            return self._stats.record_output_spec(result)
        except StopIteration:
          with timer:
            self._exhausted_iterator_state[self._next_index_in_cycle] = (
                iterator_to_use.get_state()
            )
            if self._keep_iterators_after_stop_iteration:
              self._exhausted_iterators[self._next_index_in_cycle] = (
                  self._iterators_in_use_indices[self._next_index_in_cycle],
                  iterator_to_use,
              )
            self._iterators_in_use[self._next_index_in_cycle] = None
            self._next_index_in_cycle = (
                self._next_index_in_cycle + 1
            ) % self._cycle_length
            continue
      with timer:
        if self._next_index_in_datasets < len(self._datasets):
          self._iterators_in_use[self._next_index_in_cycle] = next(
              self._prefetch_ds_iter
          )
          self._exhausted_iterator_state[self._next_index_in_cycle] = None
          self._iterators_in_use_indices[self._next_index_in_cycle] = (
              self._next_index_in_datasets
          )
          # For elastic iterators, we might have a future state saved for this
          # dataset iterator from which to resume from.
          if (
              self._next_index_in_datasets in self._future_states
              and self._iterators_in_use[self._next_index_in_cycle]
          ):
            future_state = self._future_states.pop(self._next_index_in_datasets)
            self._iterators_in_use[self._next_index_in_cycle].set_state(  # pyrefly: ignore[missing-attribute]
                future_state
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
        iterators_in_use_states[i] = it.get_state()  # pyrefly: ignore[unsupported-operation]
      elif self._exhausted_iterator_state[i] is not None:
        iterators_in_use_states[i] = self._exhausted_iterator_state[i]  # pyrefly: ignore[unsupported-operation]
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
        iterators_in_use_states[i] = it.get_state()  # pyrefly: ignore[unsupported-operation]
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
    state = {
        "next_index_in_cycle": self._next_index_in_cycle,
        "next_index_in_datasets": self._next_index_in_datasets,
        "iterators_in_use_indices": self._iterators_in_use_indices.copy(),
        "iterators_in_use_states": iterators_in_use_states,
        "exhausted": exhausted,
        "future_states": self._future_states,
    }
    return state

  def get_shard_states(self) -> Sequence[Any]:
    """Retrieves individual shard states for elastic resizing."""
    state = self.get_state()
    indices = state["iterators_in_use_indices"]
    states = state["iterators_in_use_states"]
    exhausted = state["exhausted"]
    next_index_in_datasets = state["next_index_in_datasets"]

    shard_states = [None] * len(self._datasets)

    for i in range(len(self._datasets)):  # pylint: disable=protected-access
      # `next_index_in_datasets` marks the end of the cycle. If the current
      # shard index is greater than or equal to `next_index_in_datasets`, it
      # means the current shard has not yet started to be iterated on or have a
      # future state has been saved for it.
      if i >= next_index_in_datasets:
        if i in self._future_states:
          shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
              "exhausted": 0,
              "state": self._future_states[i],
          }
        else:
          shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
              "exhausted": 0,
              "state": self._get_iterator_start_state(i),  # pylint: disable=protected-access
          }
      elif i not in indices:
        # These shards are exhausted but should still create a state to maintain
        # static state spec shapes.
        shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
            "exhausted": 1,
            "state": self._get_iterator_start_state(i),  # pylint: disable=protected-access
        }

    for index, ds_state, is_exhausted in zip(indices, states, exhausted):
      # These shards are currently being iterated on.
      shard_states[index] = {  # pyrefly: ignore[unsupported-operation]
          "exhausted": is_exhausted,
          "state": ds_state,
      }

    return shard_states

  def set_state(self, state):
    exhausted = state["exhausted"]
    for index_in_cycle, (index_in_datasets, it_state) in enumerate(
        zip(state["iterators_in_use_indices"], state["iterators_in_use_states"])
    ):
      if it_state is None:
        self._iterators_in_use[index_in_cycle] = None
      elif exhausted[index_in_cycle] == 0:
        iterator = self._iterators_in_use[index_in_cycle]
        if (
            index_in_datasets != self._iterators_in_use_indices[index_in_cycle]
            or iterator is None
        ):
          # The iterator currently in use is either exhausted or corresponds to
          # a different dataset. We need to create a new iterator or check the
          # exhausted iterators list.
          if (
              self._keep_iterators_after_stop_iteration
              and self._exhausted_iterators[index_in_cycle] is not None
              and self._exhausted_iterators[index_in_cycle][0]  # pyrefly: ignore[unsupported-operation]
              == index_in_datasets
          ):
            _, iterator = self._exhausted_iterators[index_in_cycle]  # pyrefly: ignore[not-iterable]
            self._exhausted_iterators[index_in_cycle] = None
          else:
            iterator = _add_prefetch_and_make_iterator(
                self._datasets[index_in_datasets],
                interleave_iterator=weakref.ref(self),
                start_prefetch=False,
            )
        # Only update the iterator state if it is given
        if it_state:
          iterator.set_state(it_state)
        self._iterators_in_use[index_in_cycle] = iterator
      else:
        self._exhausted_iterator_state[index_in_cycle] = it_state
        self._iterators_in_use[index_in_cycle] = None

    self._prefetch_ds_iter.set_state(
        {"next_index": state["next_index_in_datasets"]}
    )
    self._next_index_in_cycle = state["next_index_in_cycle"]
    self._next_index_in_datasets = state["next_index_in_datasets"]
    self._iterators_in_use_indices = state["iterators_in_use_indices"]
    self._future_states = cast(dict[int, Any], state.get("future_states", {}))

  def set_shard_states(self, shard_states: Sequence[Any]) -> None:
    active_states = []
    for indx, shard_state in enumerate(shard_states):
      if not shard_state["exhausted"]:
        active_states.append((indx, shard_state["state"]))

    iterators_in_use_indices = []
    iterators_in_use_states = []
    exhausted = []
    count = 0
    future_states = {}
    for indx, s in active_states:
      if count < self._cycle_length:
        iterators_in_use_indices.append(indx)
        iterators_in_use_states.append(s)
        exhausted.append(0)
        count += 1
      elif s:
        future_states[indx] = s
    next_index_in_datasets = (
        max(iterators_in_use_indices) + 1 if iterators_in_use_indices else 0
    )

    # This is the case where the cycle length is greater than the number of
    # non-exhausted shards. We will just go back through the datasets to fill
    # the cycle length.
    fill_index = 0
    while count < self._cycle_length and fill_index < len(shard_states):
      if shard_states[fill_index]["exhausted"]:
        iterators_in_use_indices.append(fill_index)
        iterators_in_use_states.append(shard_states[fill_index]["state"])
        exhausted.append(1)
        count += 1
      fill_index += 1

    new_state = {
        "next_index_in_cycle": 0,
        "next_index_in_datasets": next_index_in_datasets,
        "iterators_in_use_indices": iterators_in_use_indices,
        "iterators_in_use_states": iterators_in_use_states,
        "exhausted": exhausted,
        "future_states": future_states,
    }
    self.set_state(new_state)

  def _get_next_index(self) -> int:
    if len(self._datasets) == 1:
      it = self._iterators_in_use[0]
      if it is None:
        return 0
      return dataset.get_next_index(it)
    raise NotImplementedError(
        "get_next_index is not supported for InterleaveDatasetIterator with"
        " more than one dataset."
    )

  def _set_next_index(self, index: int) -> None:
    if len(self._datasets) == 1:
      # Ensure iterator is created by calling get_state.
      _ = self.get_state()
      it = self._iterators_in_use[0]
      assert it is not None
      dataset.set_next_index(it, index)
    else:
      raise NotImplementedError(
          "set_next_index is not supported for InterleaveDatasetIterator with"
          " more than one dataset."
      )

  def set_keep_iterators_after_stop_iteration(
      self, keep_iterators: bool
  ) -> None:
    # Determines whether the iterators should be kept alive after
    # StopIteration is raised by `__next__`. This is used by
    # `RepeatDatasetIterator` to allow for resetting the iterator state and
    # continuing iteration without recreating the iterators.
    self._keep_iterators_after_stop_iteration = keep_iterators

  def close(self) -> None:
    """Closes the iterator and shuts down the iterator prefetching."""
    if self._closed:
      return
    self._closed = True
    self._prefetch_ds_iter.close()
    for iterator in self._iterators_in_use:
      if iterator is not None:
        iterator.close()

  def start_prefetch(self):
    self._started = True
    self._prefetch_ds_iter.start_prefetch()
    for it in self._iterators_in_use:
      if it is not None:
        it.start_prefetch()

  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ) -> stats.Stats:
    # We pass an empty list of parents to `stats.make_stats` below. The
    # parents of InterleaveDatasetIterator are the iterators of the
    # datasets being interleaved. These are dynamically created and tracked
    # in `self._iterators_in_use`. When an iterator produces an element in
    # `__next__`, its Stats object is added to `self._stats._parents`.
    config = stats.StatsConfig(
        name=str(self),
        transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
        iter_weakref=stats.HashableWeakRef(self),
        node_type=stats.NodeType.INTERLEAVE,
    )
    # If the stats object has already been initialized, copy the queues from
    # the original stats object to the new stats object.
    output_spec = None
    if "_stats" in self.__dict__:
      config.stats_out_queue = self._stats._config.stats_out_queue  # pylint: disable=protected-access
      config.stats_in_queues = self._stats._config.stats_in_queues  # pylint: disable=protected-access
      # output spec is constructed while iterating through the dataset.
      output_spec = self._stats.output_spec
    self._stats = stats.make_stats(
        config,
        [],
        execution_tracking_mode=execution_tracking_mode,
    )
    self._stats._self_output_spec = output_spec  # pylint: disable=protected-access
    return self._stats

  def __str__(self) -> str:
    return (
        f"InterleaveDatasetIterator([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )

  def _get_iterator_start_state(self, index: int) -> dict[str, Any]:
    if index not in self._cached_start_states:
      it = _add_prefetch_and_make_iterator(
          self._datasets[index],
          weakref.ref(self),
          start_prefetch=False,
      )
      self._cached_start_states[index] = it.get_state()
      it.close()
    return self._cached_start_states[index]


def _add_prefetch_and_make_iterator(
    ds: dataset.IterDataset[T] | dataset.MapDataset[T],
    interleave_iterator: weakref.ref[dataset.DatasetIterator[T]],
    start_prefetch: bool,
    starting_state: dict[str, Any] | None = None,
) -> dataset.DatasetIterator[T]:
  """Adds prefetching to an IterDataset and returns an iterator.

  If the input is a MapDataset, prefetching is handled by `MapDataset.__iter__`.
  If the input is an IterDataset, a `ThreadPrefetchIterDataset` is used to
  add prefetching.

  Args:
    ds: The dataset to create an iterator from.
    interleave_iterator: The `InterleaveDatasetIterator` instance.
    start_prefetch: Whether to start the prefetching on iterator creation.
    starting_state: The state of the iterator to set.

  Returns:
    A `dataset.DatasetIterator` for the given dataset, with prefetching
    enabled if applicable.

  Raises:
    RuntimeError: If the interleave_iterator has been garbage collected.
  """
  # pylint: disable=protected-access
  interleave_iterator_obj = interleave_iterator()
  assert isinstance(
      interleave_iterator_obj,
      (InterleaveDatasetIterator, TunableInterleaveDatasetIterator),
  )
  if interleave_iterator_obj is None:
    raise RuntimeError("InterleaveDatasetIterator has been garbage collected.")
  iter_buffer_size = interleave_iterator_obj._iter_buffer_size
  ctx = interleave_iterator_obj._ctx
  # Release the strong reference before potentially slow iterator creation.
  # This prevents the worker thread from triggering parent destruction.
  del interleave_iterator_obj

  if isinstance(ds, dataset.MapDataset):
    # Prefetch is automatically added in `MapDataset.__iter__`.
    iter_dataset = ds.to_iter_dataset()
  else:
    iter_dataset = prefetch.ThreadPrefetchIterDataset(
        ds, prefetch_buffer_size=iter_buffer_size
    )
  if ctx.autotuning_enabled:
    iter_dataset = autotune.AutotuneIterDataset(
        iter_dataset,
        allow_unknown_nodes=ctx.autotuning_allow_unknown_nodes,
        create_model=False,
        warmup_state=ctx.autotuning_warmup_state,
    )
  iterator = iter_dataset.__iter__()

  # Propagate options applied after InterleaveIterDataset to the iterators that
  # are being interleaved.
  iterator._ctx.dataset_options = ctx.dataset_options.merge(iterator._ctx.dataset_options)  # pylint: disable=protected-access
  iterator._ctx.autotuning_warmup_state = (  # pylint: disable=protected-access
      ctx.autotuning_warmup_state  # pylint: disable=protected-access
  )
  iterator._ctx.autotuning_model_config_args.update(  # pylint: disable=protected-access
      ctx.autotuning_model_config_args  # pylint: disable=protected-access
  )

  if start_prefetch:
    iterator.start_prefetch()
  if starting_state is not None:
    iterator.set_state(starting_state)
  # pylint: enable=protected-access
  return iterator


class InterleaveIterDataset(dataset.IterDataset[T]):
  """Interleaves the given sequence of datasets.

  The sequence can be a `MapDataset`.

  Concurrently processes at most `cycle_length` iterators and interleaves their
  elements. If `cycle_length` is larger than the number of datasets, then the
  behavior is similar to mixing the datasets with equal proportions. If
  `cycle_length` is 1, the datasets are chained.

  This dataset can be combined with ``mp_prefetch`` to parallelize reads from
  sources that do not support random access.

  Element spec inference assumes that all input datasets have the same element
  spec.

  Example:
    Interleaving four datasets with two active iterators::

      import grain

      def make_source(start):
        return grain.MapDataset.range(start, start + 2).to_iter_dataset()

      sources = grain.MapDataset.source([0, 10, 20, 30]).map(make_source)

      print(list(sources[1]))
      # [10, 11]

      interleaved_ds = grain.experimental.InterleaveIterDataset(
          sources,
          cycle_length=2,
      )

      print(list(interleaved_ds))
      # [0, 10, 1, 11, 20, 30, 21, 31]
  """

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      *,
      cycle_length: int | grain_options.AutotuneParameter,
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

  def __iter__(self) -> dataset.DatasetIterator[T]:
    use_tunable = False
    if isinstance(self._cycle_length, grain_options.AutotuneParameter):
      use_tunable = True
    else:
      for i in range(min(len(self._datasets), self._cycle_length)):
        ds = self._datasets[i]
        options = prefetch.get_dataset_options(ds)  # pyrefly: ignore[bad-argument-type]
        if options.use_tunable_interleave:
          use_tunable = True
          break

    if use_tunable:
      return TunableInterleaveDatasetIterator(
          self._datasets,
          cycle_length=self._cycle_length,
          num_make_iter_threads=self._num_make_iter_threads,
          make_iter_buffer_size=self._make_iter_buffer_size,
          iter_buffer_size=self._iter_buffer_size,
      )
    return InterleaveDatasetIterator(
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


def _cancel_and_close_future(future: futures.Future) -> None:
  """Cancels the future and closes the iterator it returns."""

  def _close_iterator_callback(f):
    try:
      iterator = f.result()
      iterator.close()
    except Exception:  # pylint: disable=broad-except
      pass

  if future.cancel():
    return

  if not future.done():
    future.add_done_callback(_close_iterator_callback)
    return

  try:
    iterator = future.result()
    iterator.close()
  except Exception:  # pylint: disable=broad-except
    pass


class TunableInterleaveDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator for InterleaveIterDataset that supports on-the-fly tuning of cycle length.

  This iterator provides additional functionality over the standard
  InterleaveDatasetIterator by allowing the `cycle_length` to be tuned
  dynamically during iteration. If cycle length is tuned dynamically, the order
  of elements will not be deterministic, but data is guaranteed not to be
  skipped or repeated.

  How it provides this functionality:
  - It tracks the state of iterators that are pushed out of the active cycle
    when `cycle_length` is reduced, storing them in `_future_states`. This
    ensures no data is lost when the cycle shrinks.
  - It manually manages the thread pool and work queue (`_queued_iterators`)
    for background iterator creation, giving it full access to inspect and
    modify the queue of iterators not yet actively producing elements. This
    enables it to scale the prefetch buffer and worker pool up or down to
    match the desired cycle length and concurrency.
  """

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      datasets: Sequence[dataset.IterDataset[T] | dataset.MapDataset[T]],
      cycle_length: int | grain_options.AutotuneParameter,
      num_make_iter_threads: int = 1,
      make_iter_buffer_size: int = 1,
      iter_buffer_size: int = 1,
  ):
    super().__init__()
    self._datasets = datasets
    self._cycle_length, self._autotune_cycle_length = (
        autotune.get_autotune_parameter(cycle_length)
    )
    self._num_make_iter_threads = num_make_iter_threads
    self._make_iter_buffer_size = make_iter_buffer_size
    self._iter_buffer_size = iter_buffer_size
    self._started = False

    # cycle_length iterators that are in use.
    self._iterators_in_use: list[dataset.DatasetIterator[T] | None] = [
        None
    ] * self._cycle_length
    # Indices of datasets being iterated over in _iterators_in_use.
    self._iterators_in_use_indices: list[int] = [
        -1 for _ in range(self._cycle_length)
    ]
    # make_iter_buffer_size iterators that are prefetched in the background.
    self._queued_iterators: collections.deque[
        tuple[int, futures.Future[dataset.DatasetIterator[T]]]
    ] = collections.deque()
    # Executor for creating iterators and starting prefetch asynchronously.
    self._executor: futures.ThreadPoolExecutor | None = None
    # Index of the next iterator to be used in _iterators_in_use.
    self._next_index_in_cycle: int = 0
    # Index of the next iterator to be put into _iterators_in_use.
    self._next_index_in_datasets: int = 0
    # Index of the next unbuffered iterator.
    self._next_index_in_unbuffered_datasets: int = 0
    # Future states of the iterators that were being used before cycle_length
    # was reduced. If an index is in this dict, then the iterator for that
    # index is not in _iterators_in_use.
    self._future_states: dict[int, dict[str, Any]] = {}
    # Indices of the datasets that are associated with future states. If an
    # index is in this set, then the iterator for that index is not in
    # _iterators_in_use or _queued_iterators.
    self._future_states_indices: set[int] = set()
    # Placeholder state for use when datasets are exhausted. This is used for
    # Pathways Remote Python , which requires state spec to remain consistent.
    # This works when the input datasets have the same state spec.
    self._placeholder_state: dict[str, Any] | None = None
    # Flag to keep exhausted iterators around for use with `RepeatIterDataset`
    self._keep_iterators_after_stop_iteration = False
    # Keeps track of the last exhausted iterator for each slot in the cycle.
    self._exhausted_iterators: list[dataset.DatasetIterator[T] | None] = [
        None
    ] * self._cycle_length
    # Indices of the datasets that are associated with exhausted iterators.
    self._exhausted_iterators_indices: list[int] = [
        -1 for _ in range(self._cycle_length)
    ]
    # Cache for the starting states of iterators.
    self._cached_start_states: dict[int, Any] = {}

  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ) -> stats.Stats:
    # We pass an empty list of parents to `stats.make_stats` below. The
    # parents of InterleaveDatasetIterator are the iterators of the
    # datasets being interleaved. These are dynamically created and tracked
    # in `self._iterators_in_use`. When an iterator produces an element in
    # `__next__`, its Stats object is added to `self._stats._parents`.
    config = stats.StatsConfig(
        name=str(self),
        transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
        iter_weakref=stats.HashableWeakRef(self),
        node_type=stats.NodeType.INTERLEAVE,
    )
    # If the stats object has already been initialized, copy the queues from
    # the original stats object to the new stats object.
    output_spec = None
    if "_stats" in self.__dict__:
      config.stats_out_queue = self._stats._config.stats_out_queue  # pylint: disable=protected-access
      config.stats_in_queues = self._stats._config.stats_in_queues  # pylint: disable=protected-access
      # output spec is constructed while iterating through the dataset.
      output_spec = self._stats.output_spec
    self._stats = stats.make_stats(
        config,
        [],
        execution_tracking_mode=execution_tracking_mode,
    )
    self._stats._self_output_spec = output_spec  # pylint: disable=protected-access
    return self._stats

  def _increment_next_index_in_cycle(self):
    self._next_index_in_cycle = (
        self._next_index_in_cycle + 1
    ) % self._cycle_length

  def _fill_queued_iterators(self):
    """Fills the queue of iterators that are not yet in use."""
    if self._make_iter_buffer_size == 0 or self._num_make_iter_threads == 0:
      # Iterator prefetching is not possible when threads or buffer size are 0.
      return
    while len(self._queued_iterators) < self._make_iter_buffer_size:
      if (
          self._next_index_in_unbuffered_datasets >= len(self._datasets)
          and not self._future_states_indices
      ):
        break
      if self._future_states_indices:
        # Prioritize using the future states of iterators that were being used
        # before cycle_length was reduced.
        index = self._future_states_indices.pop()
        future = self._create_iterator_asynchronously(
            index, starting_state=self._future_states[index]
        )
      else:
        index = self._next_index_in_unbuffered_datasets
        starting_state = self._future_states.get(index)
        future = self._create_iterator_asynchronously(index, starting_state)
        self._next_index_in_unbuffered_datasets += 1
      self._queued_iterators.append((index, future))

  def start_prefetch(self):
    if self._started:
      return
    # Start prefetching from the iterators that are already in use.
    for it in self._iterators_in_use:
      if it is not None:
        it.start_prefetch()
    # Start prefetching iterators that are not yet in use.
    if self._num_make_iter_threads > 0 and self._make_iter_buffer_size > 0:
      assert self._executor is None
      self._executor = futures.ThreadPoolExecutor(
          self._num_make_iter_threads, thread_name_prefix="grain-interleave"
      )
      self._fill_queued_iterators()
    self._started = True

  def _stop_prefetch(self):
    if not self._started:
      return
    self._started = False

    while self._queued_iterators:
      index, future = self._queued_iterators.popleft()
      _cancel_and_close_future(future)

      if index not in self._future_states:
        self._next_index_in_unbuffered_datasets -= 1
      else:
        self._future_states_indices.add(index)
    if self._executor is not None:
      self._executor.shutdown(wait=False)
      self._executor = None

  def close(self):
    self._stop_prefetch()
    for iterator in self._iterators_in_use:
      if iterator is not None:
        iterator.close()

  def set_keep_iterators_after_stop_iteration(self, keep_iterators: bool):
    """Sets flag to keep exhausted iterators around for `RepeatIterDataset`."""
    self._keep_iterators_after_stop_iteration = keep_iterators

  def _set_cycle_length(self, new_cycle_length: int):
    # Warning: This method removes all guarantees of deterministic ordering.
    if new_cycle_length < 0:
      raise ValueError("cycle_length must be non-negative.")
    if new_cycle_length > len(self._datasets):
      new_cycle_length = len(self._datasets)
    if new_cycle_length == self._cycle_length:
      return

    if new_cycle_length < self._cycle_length:
      # Move the iterators that are no longer in use to the queued iterators.
      for _ in range(new_cycle_length, self._cycle_length):
        it = self._iterators_in_use.pop()
        index = self._iterators_in_use_indices.pop()
        self._exhausted_iterators.pop()
        self._exhausted_iterators_indices.pop()
        if it is not None:
          future = futures.Future()
          future.set_result(it)
          self._queued_iterators.appendleft((index, future))
          # Store the state of the iterator for later use.
          state = it.get_state()
          self._future_states[index] = state
      # Ensure the queue size is respected.
      while len(self._queued_iterators) > new_cycle_length:
        index, future = self._queued_iterators.pop()
        _cancel_and_close_future(future)
        if index in self._future_states:
          self._future_states_indices.add(index)
        else:
          self._next_index_in_unbuffered_datasets -= 1

    if new_cycle_length > self._cycle_length:
      # Increase the size of the lists to accommodate the new cycle length.
      self._iterators_in_use.extend(
          [None] * (new_cycle_length - self._cycle_length)
      )
      self._iterators_in_use_indices.extend(
          [-1] * (new_cycle_length - self._cycle_length)
      )
      self._exhausted_iterators.extend(
          [None] * (new_cycle_length - self._cycle_length)
      )
      self._exhausted_iterators_indices.extend(
          [-1] * (new_cycle_length - self._cycle_length)
      )

    self._cycle_length = new_cycle_length
    self._next_index_in_cycle = self._next_index_in_cycle % self._cycle_length

  def _apply_autotune_updates_if_present(self):
    if self._autotune_cycle_length is not None:
      new_cycle_length = int(round(self._autotune_cycle_length.get_value()))
      if new_cycle_length != self._cycle_length:
        logging.vlog(
            1,
            "Autotune updated TunableInterleaveDatasetIterator cycle length"
            " from %d to %d.",
            self._cycle_length,
            new_cycle_length,
        )
        self._set_cycle_length(new_cycle_length)

  def _set_iter_buffer_size(self, new_iter_buffer_size: int):
    # Subsequent iterators will have the new buffer size.
    if new_iter_buffer_size < 0:
      raise ValueError("iter_buffer_size must be non-negative.")
    self._iter_buffer_size = new_iter_buffer_size

  def _set_make_iter_buffer_size(self, new_make_iter_buffer_size: int):
    if new_make_iter_buffer_size < 0:
      raise ValueError("make_iter_buffer_size must be non-negative.")
    while len(self._queued_iterators) > new_make_iter_buffer_size:
      index, future = self._queued_iterators.pop()
      _cancel_and_close_future(future)
      if index in self._future_states:
        self._future_states_indices.add(index)
      else:
        self._next_index_in_unbuffered_datasets -= 1
    self._make_iter_buffer_size = new_make_iter_buffer_size

  def _set_num_make_iter_threads(self, new_num_make_iter_threads: int):
    if new_num_make_iter_threads < 0:
      raise ValueError("num_make_iter_threads must be non-negative.")
    self._num_make_iter_threads = new_num_make_iter_threads
    if self._make_iter_buffer_size == 0:
      return
    old_executor = self._executor
    if self._num_make_iter_threads > 0:
      self._executor = futures.ThreadPoolExecutor(
          self._num_make_iter_threads, thread_name_prefix="grain-interleave"
      )
    else:
      self._executor = None
    if old_executor is not None:
      # Allows the old executor to finish running the tasks it was already
      # assigned asynchronously.
      old_executor.shutdown(wait=False)

  def _create_iterator_synchronously(
      self,
      index: int,
      starting_state: dict[str, Any] | None = None,
      start_prefetch: bool = True,
  ) -> dataset.DatasetIterator[T]:
    return _add_prefetch_and_make_iterator(
        self._datasets[index],
        starting_state=starting_state,
        interleave_iterator=weakref.ref(self),
        start_prefetch=start_prefetch,
    )

  def _create_iterator_asynchronously(
      self, index: int, starting_state: dict[str, Any] | None = None
  ) -> futures.Future[dataset.DatasetIterator[T]]:
    if self._executor is None:
      raise ValueError("Executor has not been initialized.")
    return self._executor.submit(
        _add_prefetch_and_make_iterator,
        self._datasets[index],
        starting_state=starting_state,
        interleave_iterator=weakref.ref(self),
        start_prefetch=True,
    )

  def _get_next_iterator(
      self, start_prefetch: bool = True
  ) -> tuple[int, dataset.DatasetIterator[T]]:
    """Gets the next iterator to place in the cycle along with its index."""
    if self._queued_iterators:
      # Case of asynchronous iterator creation.
      index, future = self._queued_iterators.popleft()
      self._future_states.pop(index, None)
      self._fill_queued_iterators()
      iterator = future.result()
    else:
      # Case of synchronous iterator creation.
      index = (
          self._future_states_indices.pop()
          if self._future_states_indices
          else self._next_index_in_datasets
      )
      starting_state = self._future_states.pop(index, None)
      iterator = self._create_iterator_synchronously(
          index,
          starting_state=starting_state,
          start_prefetch=start_prefetch,
      )
      if index == self._next_index_in_datasets:
        self._next_index_in_unbuffered_datasets += 1

    if index == self._next_index_in_datasets:
      self._next_index_in_datasets += 1
    if self._placeholder_state is None:
      self._placeholder_state = iterator.get_state()
    return index, iterator

  def __next__(self) -> T:
    self._assert_not_closed()
    self.start_prefetch()
    self._apply_autotune_updates_if_present()
    while True:
      # If the slot is empty, fill it with the next iterator.
      if self._iterators_in_use[
          self._next_index_in_cycle
      ] is None and self._next_index_in_datasets < len(self._datasets):
        self._fill_queued_iterators()
        (
            self._iterators_in_use_indices[self._next_index_in_cycle],
            self._iterators_in_use[self._next_index_in_cycle],
        ) = self._get_next_iterator()
        # TODO: Optimize stats aggregation if needed.
        # pylint: disable=protected-access
        self._stats._parents = [
            it._stats for it in self._iterators_in_use if it is not None
        ]
        # pylint: enable=protected-access

      # If the slot is not empty, try to get the next element from the iterator.
      it = self._iterators_in_use[self._next_index_in_cycle]
      if it is not None:
        try:
          element = next(it)
          self._increment_next_index_in_cycle()
          return element
        except StopIteration:
          if self._keep_iterators_after_stop_iteration:
            self._exhausted_iterators[self._next_index_in_cycle] = it
            self._exhausted_iterators_indices[self._next_index_in_cycle] = (
                self._iterators_in_use_indices[self._next_index_in_cycle]
            )
          self._iterators_in_use[self._next_index_in_cycle] = None
          self._iterators_in_use_indices[self._next_index_in_cycle] = -1

      if (
          not any(self._iterators_in_use)
          and not self._queued_iterators
          and self._next_index_in_datasets >= len(self._datasets)
          and not self._future_states
      ):
        # All datasets have been exhausted.
        self._stop_prefetch()
        raise StopIteration
      self._increment_next_index_in_cycle()

  def _get_iterator_start_state(self, index: int) -> dict[str, Any]:
    if index not in self._cached_start_states:
      it = _add_prefetch_and_make_iterator(
          self._datasets[index],
          weakref.ref(self),
          start_prefetch=False,
      )
      self._cached_start_states[index] = it.get_state()
      it.close()
    return self._cached_start_states[index]

  def get_shard_states(self) -> Sequence[Any]:
    state = self.get_state()
    indices = state["iterators_in_use_indices"]
    states = state["iterators_in_use_states"]
    future_states = state["future_states"]

    shard_states = [None] * len(self._datasets)

    for i in range(len(self._datasets)):
      if i in indices:
        idx = indices.index(i)
        shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
            "exhausted": 0,
            "state": states[idx],
        }
      elif i in future_states:
        shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
            "exhausted": 0,
            "state": future_states[i],
        }
      elif i < state["next_index_in_datasets"]:
        shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
            "exhausted": 1,
            "state": self._get_iterator_start_state(i),
        }
      else:
        shard_states[i] = {  # pyrefly: ignore[unsupported-operation]
            "exhausted": 0,
            "state": self._get_iterator_start_state(i),
        }

    return shard_states

  def set_shard_states(self, shard_states: Sequence[Any]) -> None:
    active_states = []
    for indx, shard_state in enumerate(shard_states):
      if not shard_state["exhausted"]:
        active_states.append((indx, shard_state["state"]))

    iterators_in_use_indices = []
    iterators_in_use_states = []
    exhausted = []
    count = 0
    future_states = {}

    target_cycle_length = self._cycle_length

    for indx, state in active_states:
      if count < target_cycle_length:
        iterators_in_use_indices.append(indx)
        iterators_in_use_states.append(state)
        exhausted.append(0)
        count += 1
      else:
        future_states[indx] = state

    next_index_in_datasets = (
        max(iterators_in_use_indices) + 1 if iterators_in_use_indices else 0
    )

    # Similar to what we do in InterleaveIterDataset, we fill in the
    # rest of the cycle_length from the start.
    fill_index = 0
    while count < target_cycle_length and fill_index < len(self._datasets):
      if shard_states[fill_index]["exhausted"]:
        iterators_in_use_indices.append(fill_index)
        iterators_in_use_states.append(shard_states[fill_index]["state"])
        exhausted.append(1)
        count += 1
      fill_index += 1

    new_state = {
        "cycle_length": target_cycle_length,
        "iter_buffer_size": self._iter_buffer_size,
        "make_iter_buffer_size": self._make_iter_buffer_size,
        "num_make_iter_threads": self._num_make_iter_threads,
        "next_index_in_cycle": 0,
        "next_index_in_datasets": next_index_in_datasets,
        "iterators_in_use_indices": iterators_in_use_indices,
        "iterators_in_use_states": iterators_in_use_states,
        "future_states": future_states,
        "placeholder_state": (
            self._placeholder_state
            if self._placeholder_state is not None
            else self._get_iterator_start_state(0)
        ),
    }
    self.set_state(new_state)

  def get_state(self):
    if self._placeholder_state is None:
      # This placeholder state allows state spec to remain consistent for
      # Pathways Remote Python. Populate it with the state of the first
      # iterator if next has not been called yet.
      index, it = self._get_next_iterator(start_prefetch=self._started)
      self._iterators_in_use[0] = it
      self._iterators_in_use_indices[0] = index
    iterators_in_use_states = [
        it.get_state() if it is not None else self._placeholder_state
        for it in self._iterators_in_use
    ]
    # Create iterators for the first cycle_length datasets to prevent setting
    # state from restarting processes in the case of `multiprocess_prefetch`.
    for i in range(min(self._cycle_length, len(self._datasets))):
      if (
          self._iterators_in_use[i] is None
          and self._next_index_in_datasets == i
      ):
        self._iterators_in_use_indices[i], self._iterators_in_use[i] = (
            self._get_next_iterator(start_prefetch=self._started)
        )
    return {
        "cycle_length": self._cycle_length,
        "iter_buffer_size": self._iter_buffer_size,
        "make_iter_buffer_size": self._make_iter_buffer_size,
        "num_make_iter_threads": self._num_make_iter_threads,
        "next_index_in_cycle": self._next_index_in_cycle,
        "next_index_in_datasets": self._next_index_in_datasets,
        "iterators_in_use_indices": self._iterators_in_use_indices.copy(),
        "future_states": copy.deepcopy(self._future_states),
        "iterators_in_use_states": iterators_in_use_states,
        "placeholder_state": self._placeholder_state,
    }

  def set_state(self, state) -> None:
    # Resize before setting state to avoid issues with mismatched list sizes
    # or mismatched thread count.
    if self._cycle_length != state["cycle_length"]:
      self._set_cycle_length(state["cycle_length"])
    if self._iter_buffer_size != state["iter_buffer_size"]:
      self._set_iter_buffer_size(state["iter_buffer_size"])
    if self._make_iter_buffer_size != state["make_iter_buffer_size"]:
      self._set_make_iter_buffer_size(state["make_iter_buffer_size"])
    if self._num_make_iter_threads != state["num_make_iter_threads"]:
      self._set_num_make_iter_threads(state["num_make_iter_threads"])

    for i, iterator_state in enumerate(state["iterators_in_use_states"]):
      if state["iterators_in_use_indices"][i] != -1:
        if (
            state["iterators_in_use_indices"][i]
            == self._iterators_in_use_indices[i]
        ):
          # The iterator currently in use is the same on specified in the state.
          # We can set the state of the iterator without recreating it.
          it = self._iterators_in_use[i]
          assert it is not None
          it.set_state(iterator_state)
        else:
          # The iterator currently in use is different from the one specified
          # in the state. We need to recreate the iterator.
          old_it = self._iterators_in_use[i]
          if old_it is not None:
            old_it.close()
          index = state["iterators_in_use_indices"][i]
          if index == self._exhausted_iterators_indices[i]:
            it = self._exhausted_iterators[i]
            assert it is not None, "Exhausted iterator cannot be None"
          else:
            it = self._create_iterator_synchronously(index)
          it.set_state(iterator_state)
          self._iterators_in_use[i] = it
      else:
        old_it = self._iterators_in_use[i]
        if old_it is not None:
          old_it.close()
        self._iterators_in_use[i] = None
      self._exhausted_iterators[i] = None
      self._exhausted_iterators_indices[i] = -1

    # The queued iterators are no longer valid. We need to cancel them.
    while self._queued_iterators:
      _, future = self._queued_iterators.popleft()
      _cancel_and_close_future(future)

    self._next_index_in_cycle = state["next_index_in_cycle"]
    self._next_index_in_datasets = state["next_index_in_datasets"]
    self._iterators_in_use_indices = state["iterators_in_use_indices"].copy()
    self._future_states = copy.deepcopy(state["future_states"])
    self._placeholder_state = copy.deepcopy(state["placeholder_state"])

    self._next_index_in_unbuffered_datasets = self._next_index_in_datasets
    self._future_states_indices = set(
        k
        for k in self._future_states.keys()
        if k < self._next_index_in_datasets
    )

  def _get_next_index(self) -> int:
    if len(self._datasets) == 1:
      it = self._iterators_in_use[0]
      if it is None:
        return 0
      return dataset.get_next_index(it)
    raise NotImplementedError(
        "get_next_index is not supported for TunableInterleaveDatasetIterator"
        " with more than one dataset."
    )

  def _set_next_index(self, index: int) -> None:
    if len(self._datasets) == 1:
      it = self._iterators_in_use[0]
      if it is None:
        it = self._create_iterator_synchronously(0)
        self._iterators_in_use[0] = it
      dataset.set_next_index(it, index)
    else:
      raise NotImplementedError(
          "set_next_index is not supported for TunableInterleaveDatasetIterator"
          " with more than one dataset."
      )

  def __str__(self) -> str:
    return (
        f"TunableInterleaveDatasetIterator([{len(self._datasets)} datasets],"
        f" cycle_length={self._cycle_length})"
    )
