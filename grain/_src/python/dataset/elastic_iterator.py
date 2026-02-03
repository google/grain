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
"""Iterator supporting changes in the number of hosts (dataset shards)."""

import copy
import functools
from typing import Any, TypeVar

from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import (
    filter as filter_dataset,
)
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import prefetch

T = TypeVar("T")

_GLOBAL_NEXT_INDEX_STATE_KEY = "global_next_index"


def _find_interleave_dataset(
    ds: dataset.IterDataset,
) -> interleave.InterleaveIterDataset | None:
  if isinstance(ds, interleave.InterleaveIterDataset):
    return ds
  if hasattr(ds, "parents"):
    for parent in ds.parents:
      if isinstance(parent, dataset.IterDataset):
        found = _find_interleave_dataset(parent)
        if found:
          return found
  return None


def _find_interleave_iterator(
    it: dataset.DatasetIterator,
) -> interleave.InterleaveDatasetIterator | None:
  if isinstance(it, interleave.InterleaveDatasetIterator):
    return it
  if hasattr(it, "_parents"):
    for parent in it._parents:  # pylint: disable=protected-access
      found = _find_interleave_iterator(parent)
      if found:
        return found
  return None


class ElasticIterDatasetIterator(dataset.DatasetIterator):
  """Elastic iterator for InterleaveIterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      shard_options: sharding.ShardOptions,
      global_batch_size: int,
      read_options: options.ReadOptions,
  ):
    super().__init__()
    self._ds = parent
    interleave_ds = _find_interleave_dataset(parent)
    if interleave_ds is None:
      raise ValueError(
          "ElasticIterDatasetIterator requires an InterleaveIterDataset in the"
          " dataset graph."
      )
    self._cycle_length = interleave_ds._cycle_length

    self._global_batch_size = global_batch_size
    self._shard_options = shard_options
    self._read_options = read_options

    # These will be initialized when the iterator is created.
    self._iterator_started = False
    self._closed = False

  @property
  def shard_options(self) -> sharding.ShardOptions:
    return self._shard_options

  def close(self):
    if self._closed:
      return
    self._closed = True
    if "_iterator" in self.__dict__:
      self._iterator.close()

  @functools.cached_property
  def _iterator(self) -> dataset.DatasetIterator:
    ds = self._ds
    self._iterator_started = True
    if self._global_batch_size > 0:
      host_batch_size, remainder = divmod(
          self._global_batch_size, self._shard_options.shard_count
      )
      if remainder:
        raise ValueError(
            f"Global batch size {self._global_batch_size} is not divisible by"
            f" shard count {self._shard_options.shard_count}."
        )
      ds = ds.batch(host_batch_size, drop_remainder=False)

    return ds.__iter__()

  def __next__(self) -> Any:
    return next(self._iterator)

  def get_state(self):
    interleave_iter = _find_interleave_iterator(self._iterator)
    if interleave_iter is None:
      raise ValueError("Could not find InterleaveDatasetIterator.")
    state = interleave_iter.get_state()
    ds_iterator_states = {}

    indices = state["iterators_in_use_indices"]
    states = state["iterators_in_use_states"]
    exhausted = state["exhausted"]
    next_index_in_datasets = state["next_index_in_datasets"]
    for i in range(len(interleave_iter._datasets)):  # pylint: disable=protected-access
      shard_index = (
          i * self._shard_options.shard_count + self._shard_options.shard_index
      )
      # If the current shard index is greater than or equal to the next
      # index in datasets, it means the current shard has not yet started
      # to be iterated on.
      if i >= next_index_in_datasets:
        ds_iterator_states[shard_index] = {
            "exhausted": 0,
            "state": interleave_iter._get_iterator_start_state(i),  # pylint: disable=protected-access
        }
      elif i not in indices:
        # These shards are exhausted but should still create a state to maintain
        # static state spec shapes.
        ds_iterator_states[shard_index] = {
            "exhausted": 1,
            "state": interleave_iter._get_iterator_start_state(i),  # pylint: disable=protected-access
        }

    for index, ds_state, is_exhausted in zip(indices, states, exhausted):
      # These shards are currently being iterated on.
      shard_index = (
          index * self._shard_options.shard_count
          + self._shard_options.shard_index
      )
      ds_iterator_states[shard_index] = {
          "exhausted": is_exhausted,
          "state": ds_state,
      }

    return {
        "ds_iterator_states": ds_iterator_states,
        "next_index_in_cycle": state["next_index_in_cycle"],
        "iterators_in_use_indices": state["iterators_in_use_indices"],
        "next_index_in_datasets": state["next_index_in_datasets"],
        "shard_count": self._shard_options.shard_count,
    }

  def set_state(self, state):
    """Sets state by reconstructing the state for the underlying interleave."""
    ds_iterator_states = state["ds_iterator_states"]
    local_states = {}
    for shard_index, shard_state in ds_iterator_states.items():
      if (
          shard_index - self._shard_options.shard_index
      ) % self._shard_options.shard_count == 0:
        local_index = shard_index // self._shard_options.shard_count
        local_states[local_index] = shard_state

    if state.get("shard_count") == self._shard_options.shard_count:
      saved_indices = state["iterators_in_use_indices"]
      iterators_in_use_states = []
      exhausted = []
      for ind in saved_indices:
        if ind in local_states:
          shard_state = local_states[ind]
          iterators_in_use_states.append(shard_state["state"])
          exhausted.append(shard_state["exhausted"])
        else:
          iterators_in_use_states.append(None)
          exhausted.append(1)

      new_state = {
          "next_index_in_cycle": state.get("next_index_in_cycle", 0),
          "next_index_in_datasets": state["next_index_in_datasets"],
          "iterators_in_use_indices": saved_indices,
          "iterators_in_use_states": iterators_in_use_states,
          "exhausted": exhausted,
          "future_states": state.get("future_states", {}),
      }
    else:
      active_states = []
      for ind, shard_state in sorted(local_states.items()):
        if not shard_state["exhausted"]:
          active_states.append((ind, shard_state["state"]))

      iterators_in_use_indices = []
      iterators_in_use_states = []
      exhausted = []
      count = 0
      future_states = {}
      for ind, s in active_states:
        if count < self._cycle_length:
          iterators_in_use_indices.append(ind)
          iterators_in_use_states.append(s)
          exhausted.append(0)
          count += 1
        elif s:
          future_states[ind] = s
      next_index_in_datasets = (
          max(iterators_in_use_indices) + 1 if iterators_in_use_indices else 0
      )
      while count < self._cycle_length:
        iterators_in_use_indices.append(next_index_in_datasets)
        iterators_in_use_states.append(None)
        exhausted.append(1)
        count += 1

      new_state = {
          "next_index_in_cycle": 0,
          "next_index_in_datasets": next_index_in_datasets,
          "iterators_in_use_indices": iterators_in_use_indices,
          "iterators_in_use_states": iterators_in_use_states,
          "exhausted": exhausted,
          "future_states": future_states,
      }
    if "_iterator" in self.__dict__:
      self.__dict__["_iterator"].close()
    self.__dict__.pop("_iterator", None)
    interleave_iter = _find_interleave_iterator(self._iterator)
    if interleave_iter is None:
      raise ValueError("Could not find InterleaveDatasetIterator.")
    interleave_iter.set_state(new_state)


class _ElasticMapDatasetIterator(dataset.DatasetIterator):
  """Iterator for MapDatasets in ElasticIterator."""

  def __init__(
      self,
      ds: dataset.MapDataset,
      shard_options: sharding.ShardOptions,
      global_batch_size: int,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    super().__init__()
    self._ds = ds
    self._shard_options = shard_options
    self._global_batch_size = global_batch_size
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options
    self._global_next_index = 0
    self._closed = False

  @functools.cached_property
  def _iterator(self) -> dataset.DatasetIterator:
    ds = self._ds[
        self._global_next_index
        + self._shard_options.shard_index :: self._shard_options.shard_count
    ]
    host_batch_size, remainder = divmod(
        self._global_batch_size, self._shard_options.shard_count
    )
    if remainder:
      raise ValueError(
          f"Global batch size {self._global_batch_size} is not divisible by"
          f" shard count {self._shard_options.shard_count}."
      )
    ds = ds.batch(host_batch_size, drop_remainder=True)
    ds = ds.to_iter_dataset(read_options=self._read_options)
    if self._multiprocessing_options is not None:
      ds = ds.mp_prefetch(self._multiprocessing_options)
    return ds.__iter__()

  def __next__(self) -> Any:
    result = next(self._iterator)
    self._global_next_index += self._global_batch_size
    return result

  def get_state(self) -> dict[str, Any]:
    return {
        _GLOBAL_NEXT_INDEX_STATE_KEY: self._global_next_index,
    }

  def close(self):
    if self._closed:
      return
    self._closed = True
    if "_iterator" in self.__dict__:
      self._iterator.close()

  def set_state(self, state):
    self._global_next_index = state[_GLOBAL_NEXT_INDEX_STATE_KEY]
    if "_iterator" in self.__dict__:
      self.__dict__["_iterator"].close()
    self.__dict__.pop("_iterator", None)


class ElasticIterator(dataset.IterDataset):
  """Iterator supporting recovery from a checkpoint after changes in sharding.

  The input dataset is expected to be unbatched and unsharded. In order to
  provide elasticity guarantee this iterator includes both, batching and
  sharding. The iterator supports elastic re-configuration by having each
  shard produce the same exact checkpoint (while producing different data) as
  long as they are advanced the same number of steps.

  State of any shard can be used to restore the state of all of the shards after
  changes in sharding and global batch size.

  This iterator explicitly disallows many-to-one transformations without
  a fixed ratio, like `filter` and generic `IterDataset` transformations.

  IterDatasets have a few more limitations:
  - Does not guarantee determinism between scaling.
  - The limit of parallelism is the number of shards.
  - Currently doesn't support multiprocessing.
  """

  def __init__(
      self,
      ds: dataset.MapDataset | dataset.IterDataset,
      global_batch_size: int,
      shard_options: sharding.ShardOptions,
      *,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    super().__init__()
    to_check = [ds]
    while to_check:
      next_ds = to_check.pop()
      if isinstance(next_ds, filter_dataset.FilterMapDataset):
        raise ValueError(
            "ElasticIterator does not support `filter` transformation."
        )
      to_check.extend(next_ds.parents)

    self._shard_options = shard_options
    self._global_batch_size = global_batch_size
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

    if isinstance(ds, dataset.IterDataset):
      if multiprocessing_options:
        raise NotImplementedError(
            "ElasticIterator does not support multiprocessing_options for"
            " IterDataset."
        )
      # We must set the slice on the original dataset so that the interleave
      # iterator is created with the correct (sliced) datasets.
      self._ds = copy.deepcopy(ds)
      prefetch._set_slice_iter_dataset(
          self._ds,
          slice(shard_options.shard_index, None, shard_options.shard_count),
      )
      if _find_interleave_dataset(self._ds) is None:
        if self._read_options.num_threads > 1:
          datasets = []
          for i in range(self._read_options.num_threads):
            d = copy.deepcopy(self._ds)
            prefetch._set_slice_iter_dataset(
                d, slice(i, None, self._read_options.num_threads)
            )
            datasets.append(d)
          self._ds = interleave.InterleaveIterDataset(
              datasets, cycle_length=len(datasets)
          )
    else:
      self._ds = ds

  @property
  def shard_options(self) -> sharding.ShardOptions:
    return self._shard_options

  def __iter__(self) -> dataset.DatasetIterator:
    if isinstance(self._ds, dataset.IterDataset):
      return ElasticIterDatasetIterator(
          self._ds,
          self._shard_options,
          self._global_batch_size,
          self._read_options,
      )
    else:
      return _ElasticMapDatasetIterator(
          self._ds,
          self._shard_options,
          self._global_batch_size,
          self._read_options,
          self._multiprocessing_options,
      )
