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
from typing import Any, TypeVar, cast

from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import (
    filter as filter_dataset,
)
from grain._src.python.dataset.transformations import interleave

T = TypeVar("T")

_GLOBAL_NEXT_INDEX_STATE_KEY = "global_next_index"


class ElasticIterDatasetIterator(dataset.DatasetIterator):
  """Elastic iterator for InterleaveIterDatasets."""

  def __init__(
      self,
      interleave_dataset: interleave.InterleaveIterDataset,
      shard_options: sharding.ShardOptions,
      global_batch_size: int,
      drop_remainder: bool,
      read_options: options.ReadOptions,
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    # We must set the slice on the original dataset so that the interleave
    # iterator is created with the correct (sliced) datasets.
    self._ds: interleave.InterleaveIterDataset = copy.deepcopy(
        interleave_dataset
    )
    self._num_dataset_shards = len(interleave_dataset._datasets)  # pylint: disable=protected-access
    self._ds.set_slice(
        slice(shard_options.shard_index, None, shard_options.shard_count)
    )
    self._num_host_shards = len(self._ds._datasets)  # pylint: disable=protected-access
    self._cycle_length = self._ds._cycle_length  # pylint: disable=protected-access

    self._global_batch_size = global_batch_size
    self._drop_remainder = drop_remainder
    self._shard_options = shard_options
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

    # These will be initialized when the iterator is created.
    self._iterator_started = False
    self._is_batched = False
    self._closed = False

  @property
  def num_dataset_shards(self) -> int:
    return self._num_dataset_shards

  @property
  def num_host_shards(self) -> int:
    return self._num_host_shards

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
      ds = ds.batch(
          self._global_batch_size, drop_remainder=self._drop_remainder
      )
      self._is_batched = True
    if self._multiprocessing_options:
      self._prefetch_wrapped = True
      # ds = ds.mp_prefetch(self._multiprocessing_options)
    return ds.__iter__()

  def __next__(self) -> Any:
    return next(self._iterator)

  def get_state(self):
    state = self._iterator.get_state()
    ds_iterator_states = {}

    indices = state["iterators_in_use_indices"]
    states = state["iterators_in_use_states"]
    exhausted = state["exhausted"]
    next_index_in_datasets = state["next_index_in_datasets"]
    if self._is_batched:
      interleave_iter = cast(interleave.InterleaveDatasetIterator, self._iterator._parent)  # pylint: disable=protected-access
    else:
      interleave_iter = cast(
          interleave.InterleaveDatasetIterator, self._iterator
      )
    for i in range(self._num_host_shards):
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

    for index, state, is_exhausted in zip(indices, states, exhausted):
      # These shards are currently being iterated on.
      shard_index = (
          index * self._shard_options.shard_count
          + self._shard_options.shard_index
      )
      ds_iterator_states[shard_index] = {
          "exhausted": is_exhausted,
          "state": state,
      }

    return {
        "ds_iterator_states": ds_iterator_states,
    }

  def set_state(self, state):
    """Sets state by reconstructing the state for the underlying interleave."""
    ds_iterator_states = state["ds_iterator_states"]
    active_states = []

    for shard_index, shard_state in sorted(ds_iterator_states.items()):
      # Check if this state belongs to the current shard.
      if (
          shard_index - self._shard_options.shard_index
      ) % self._shard_options.shard_count == 0:
        slice_index = shard_index // self._shard_options.shard_count
        if not shard_state["exhausted"]:
          active_states.append((slice_index, shard_state["state"]))

    iterators_in_use_indices = []
    iterators_in_use_states = []
    exhausted = []
    count = 0
    future_states = {}
    for ind, state in active_states:
      if count < self._cycle_length:
        iterators_in_use_indices.append(ind)
        iterators_in_use_states.append(state)
        exhausted.append(0)
        count += 1
      elif state:
        # If a state exists for this iterator add it to future states
        future_states[ind] = state
    next_index_in_datasets = max(iterators_in_use_indices) + 1
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
    self._iterator.set_state(new_state)


class _ElasticMapDatasetIterator(dataset.DatasetIterator):
  """Iterator for MapDatasets in ElasticIterator."""

  def __init__(
      self,
      ds: dataset.MapDataset,
      shard_options: sharding.ShardOptions,
      global_batch_size: int,
      drop_remainder: bool,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    self._ds = ds
    self._shard_options = shard_options
    self._global_batch_size = global_batch_size
    self._drop_remainder = drop_remainder
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options
    self._global_next_index = 0
    self._closed = False

  @functools.cached_property
  def _iterator(self):
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
    if self._multiprocessing_options:
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


class ElasticIterDataset(dataset.IterDataset):
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
  """

  def __init__(
      self,
      parents: dataset.MapDataset | dataset.IterDataset,
      shard_options: sharding.ShardOptions,
      *,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
      drop_remainder: bool = False,
      global_batch_size: int = 0,
  ):
    super().__init__()
    self.num_dataset_shards = 0
    self._ds = parents
    if isinstance(parents, dataset.IterDataset):
      if not isinstance(parents, interleave.InterleaveIterDataset):
        raise ValueError(
            "ElasticIterator only supports sliceable InterleaveIterDataset"
        )
      self.num_dataset_shards = len(parents._datasets)  # pylint: disable=protected-access
    else:
      to_check = [parents]
      while to_check:
        next_ds = to_check.pop()
        if isinstance(next_ds, filter_dataset.FilterMapDataset):
          raise ValueError(
              "ElasticIterDataset does not support `filter` transformation."
          )
        to_check.extend(next_ds.parents)

    self._shard_options = shard_options
    self._global_batch_size = global_batch_size
    self._drop_remainder = drop_remainder
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

  @property
  def shard_options(self) -> sharding.ShardOptions:
    return self._shard_options

  def __iter__(self) -> dataset.DatasetIterator:
    if isinstance(self._ds, interleave.InterleaveIterDataset):
      return ElasticIterDatasetIterator(
          self._ds,
          self._shard_options,
          self._global_batch_size,
          self._drop_remainder,
          self._read_options,
          self._multiprocessing_options,
      )
    else:
      return _ElasticMapDatasetIterator(
          self._ds,
          self._shard_options,
          self._global_batch_size,
          self._drop_remainder,
          self._read_options,
          self._multiprocessing_options,
      )
