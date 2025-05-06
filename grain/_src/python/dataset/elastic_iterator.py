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
from typing import Any

from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import (
    filter as filter_dataset,
)

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
