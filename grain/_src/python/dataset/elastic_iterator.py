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

import functools
from typing import Any, Sequence, TypeVar

from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import (
    filter as filter_dataset,
)
from grain._src.python.dataset.transformations import interleave

T = TypeVar("T")

_GLOBAL_NEXT_INDEX_STATE_KEY = "global_next_index"


class ElasticIterator(dataset.DatasetIterator):
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
      ds: dataset.MapDataset,
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
    self._ds = ds
    self._global_batch_size = global_batch_size
    self._shard_options = shard_options
    self._global_next_index = 0
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

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

  def __iter__(self) -> dataset.DatasetIterator:
    return self

  def __next__(self) -> Any:
    result = next(self._iterator)
    self._global_next_index += self._global_batch_size
    return result

  def get_state(self) -> dict[str, Any]:
    return {
        _GLOBAL_NEXT_INDEX_STATE_KEY: self._global_next_index,
    }

  def set_state(self, state: dict[str, Any]):
    self._global_next_index = state[_GLOBAL_NEXT_INDEX_STATE_KEY]
    # Reset the iterator if it was already created.
    self.__dict__.pop("_iterator", None)


class ElasticIterDatasetIterator(dataset.DatasetIterator):
  """Iterator for ElasticIterDataset.

  This class acts as a wrapper around InterleaveDatasetIterator, applying
  sharding and batching dynamically to the datasets. Typically, sharded datasets
  can not be resharded and distributed to iterators. This class
  provides a way to do this by taking in the maximum number of dataset shards
  and interleaving those shards into a variable number of iterators.

  Caveats:
    - Order of elements is not guaranteed.

  Usage:
    parquet_files = ep.glob("/path/to/some/files/*.parquet")
    ds = [
        ParquetIterDataset(f) for f in parquet_files
    ]
    it = ElasticIterDatasetIterator(
        ds,
        shard_options=sharding.ShardOptions(shard_index=jax.process_id(),
          shard_count=10),
        global_batch_size=3,
    )
    iterator = iter(it)
    x = next(iterator)

    # Continue to use the iterator as usual and save it to a checkpoint with the
    # dedicated elastic checkpoint API.
    elastic_checkpoint.save_elastic_iterator(temp_dir, it)

    # When restoring, the number of processes can be changed and elastic
    # iterator will be restored accordingly.
    it = ElasticIterDatasetIterator(
        ds,
        shard_options=sharding.ShardOptions(shard_index=jax.process_id(),
          shard_count=20),
        global_batch_size=3,
    )
    elastic_checkpoint.restore_elastic_iterator(temp_dir, it)
  """

  def __init__(
      self,
      ds: Sequence[dataset.IterDataset],
      shard_options: sharding.ShardOptions,
      global_batch_size: int,
      *,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
      cycle_length: int | None = None,
      num_make_iter_threads: int = 1,
      make_iter_buffer_size: int = 1,
      iter_buffer_size: int = 1,
  ):
    super().__init__()
    self._ds = ds
    self._global_batch_size = global_batch_size
    self._shard_options = shard_options
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

    # InterleaveDatasetIterator options.
    self._cycle_length = cycle_length or global_batch_size
    self._num_make_iter_threads = num_make_iter_threads
    self._make_iter_buffer_size = make_iter_buffer_size
    self._iter_buffer_size = iter_buffer_size

    self._total_num_shards = len(ds)
    # The shard indices that are assigned to this iterator.
    self._shard_indices = list(
        range(
            self._shard_options.shard_index,
            self._total_num_shards,
            self._shard_options.shard_count,
        )
    )
    # The corresponding iterators for each shard index.
    if self._global_batch_size == 1:
      self._ds_iterators = [
          ds.__iter__()
          for ds in self._ds[
              self._shard_options.shard_index :: self._shard_options.shard_count
          ]
      ]
    else:
      self._ds_iterators = [
          ds.batch(self._global_batch_size, drop_remainder=True).__iter__()
          for ds in self._ds[
              self._shard_options.shard_index :: self._shard_options.shard_count
          ]
      ]

  @property
  def shard_options(self) -> sharding.ShardOptions:
    return self._shard_options

  @property
  def total_num_shards(self) -> int:
    return self._total_num_shards

  @functools.cached_property
  def _iterator(self) -> dataset.DatasetIterator:
    return interleave.InterleaveDatasetIterator(
        self._ds_iterators,
        cycle_length=self._cycle_length,
        num_make_iter_threads=self._num_make_iter_threads,
        make_iter_buffer_size=self._make_iter_buffer_size,
        iter_buffer_size=self._iter_buffer_size,
    )

  def __iter__(self) -> dataset.DatasetIterator:
    return self

  def __next__(self) -> Any:
    return next(self._iterator)

  def get_state(self) -> dict[str, Any]:
    host_iterator_states = {
        indx: it.get_state()
        for indx, it in zip(self._shard_indices, self._ds_iterators)
    }
    state = {
        "total_num_shards": self._total_num_shards,
    }
    state["ds_iterator_states"] = host_iterator_states
    return state

  def set_state(self, state: dict[str, Any]):
    saved_iterator_states = state["ds_iterator_states"]
    for k, v in saved_iterator_states.items():
      indx = k // self.shard_options.shard_count
      self._ds_iterators[indx].set_state(v)
    self.__dict__.pop("_iterator", None)

  def update_shard_iterator_state(
      self, shard_index: int, state: dict[str, Any]
  ):
    if shard_index not in self._shard_indices:
      # This should never happen.
      raise ValueError(
          f"Shard index {shard_index} is not in the shard indices"
          f" {self._shard_indices}."
      )
    indx = shard_index // self.shard_options.shard_count
    self._ds_iterators[indx].set_state(state)
