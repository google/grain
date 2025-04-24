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
"""A sampler is reponsible for providing which data records to load next."""

from typing import Optional, Protocol

from grain._src.core import monitoring as grain_monitoring
from grain._src.core import sharding
from grain._src.python import record
from grain._src.python.dataset import dataset
import numpy as np

from grain._src.core import monitoring


_api_usage_counter = monitoring.Counter(
    "/grain/python/samplers/api",
    metadata=monitoring.Metadata(
        description="Sampler API initialization counter."
    ),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)


class Sampler(Protocol):
  """Interface for PyGrain-compatible sampler."""

  def __getitem__(self, index: int) -> record.RecordMetadata:
    """Returns the RecordMetadata for a global index."""


class SequentialSampler:
  """Basic sampler implementation that provides records in order."""

  def __init__(
      self,
      num_records: int,
      shard_options: sharding.ShardOptions = sharding.NoSharding(),
      seed: Optional[int] = None,
  ):
    if num_records <= 0:
      raise ValueError(
          "Invalid number of records in Sampler. "
          f"Got {num_records} records, but number of records "
          "must be greater than 0."
      )
    self._num_records = num_records
    self._shard_options = shard_options
    if shard_options.drop_remainder:
      num_records_per_shard = self._num_records // shard_options.shard_count
      self._max_index = num_records_per_shard * shard_options.shard_count
    else:
      self._max_index = self._num_records
    self._seed = seed
    _api_usage_counter.Increment("SequentialSampler")

  def __repr__(self) -> str:
    return (
        f"SequentialSampler(num_records={self._num_records}, "
        f"shard_options={self._shard_options!r})"
    )

  def __getitem__(self, index: int) -> record.RecordMetadata:
    if index < 0 or index >= self._max_index:
      raise IndexError(
          f"RecordMetadata object index is out of bounds; Got index {index},"
          f" allowed indices should be in [0, {self._max_index}]"
      )
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    return record.RecordMetadata(index=index, record_key=index, rng=rng)


class _ShardMapDataset(dataset.MapDataset):
  """Shards the parent into consecutive pieces."""

  def __init__(
      self, parent: dataset.MapDataset, shard_options: sharding.ShardOptions
  ):
    super().__init__(parent)
    self._start, self._end = sharding.even_split(
        len(self._parent), shard_options
    )

  def __len__(self) -> int:
    return self._end - self._start

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    epoch = index // len(self)
    index_in_epoch = index % len(self)
    index = epoch * len(self._parent) + index_in_epoch + self._start
    return self._parent[index]


class IndexSampler:
  """Base index sampler for training on a single datasource.

  This index sampler supports the following operations:
  - Sharding of the dataset.
  - Global shuffle of the dataset.
  """

  def __init__(
      self,
      num_records: int,
      shard_options: sharding.ShardOptions = sharding.NoSharding(),
      shuffle: bool = False,
      num_epochs: Optional[int] = None,
      seed: Optional[int] = None,
  ):
    if num_records <= 0:
      raise ValueError(
          "Invalid number of records in Sampler. "
          f"Got {num_records} records, but number of records "
          "must be greater than 0."
      )
    if num_epochs is not None and num_epochs <= 0:
      raise ValueError(
          "Invalid number of epochs in Index Sampler."
          f"Got {num_epochs} epochs, but number of epochs "
          "must be greater than 0."
      )
    if shuffle and seed is None:
      raise ValueError("Shuffling requires specifying a seed.")

    if shuffle and not isinstance(seed, int):
      raise TypeError(
          f"Expected seed of int type. Got seed with type {type(seed)}"
      )

    if seed is not None:
      if seed < 0 or seed.bit_length() > 32:
        raise ValueError("Seed should be positive 32-bit integer.")

    self._num_records = num_records
    self._shard_options = shard_options
    self._shuffle = shuffle
    self._num_epochs = num_epochs
    self._seed = seed
    self._max_index = None if num_epochs is None else num_epochs * num_records

    self._record_keys = dataset.MapDataset.range(num_records)
    if not isinstance(shard_options, sharding.NoSharding):
      self._record_keys = _ShardMapDataset(self._record_keys, shard_options)
      if self._max_index is not None and shard_options.drop_remainder:
        self._max_index = min(  # Account for no remainder
            self._max_index,
            len(self._record_keys) * shard_options.shard_count * num_epochs,
        )
    if shuffle:
      self._record_keys = self._record_keys.shuffle(seed=seed)
    _api_usage_counter.Increment("IndexSampler")

  def __repr__(self) -> str:
    return (
        f"IndexSampler(num_records={self._num_records}, "
        f"shard_options={self._shard_options!r}, "
        f"shuffle={self._shuffle}, "
        f"num_epochs={self._num_epochs}, "
        f"seed={self._seed})"
    )

  def __getitem__(self, index: int) -> record.RecordMetadata:
    if index < 0 or (self._max_index is not None and index >= self._max_index):
      raise IndexError(
          f"RecordMetadata object index is out of bounds; Got index {index},"
          f" allowed indices should be in [0, {self._max_index}]"
      )
    record_key = self._record_keys[index // self._shard_options.shard_count]
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    next_record = record.RecordMetadata(
        index=index, record_key=record_key, rng=rng
    )
    return next_record
