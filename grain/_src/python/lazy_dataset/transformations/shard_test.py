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
"""Tests for shard transformation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import sharding
from grain._src.python.lazy_dataset import lazy_dataset
import grain._src.python.lazy_dataset.transformations.shard as shard_ds


class ShardLazyMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data_len = 20
    self.range_ds = lazy_dataset.RangeLazyMapDataset(self.data_len)
    self.range_py_list = list(range(self.data_len))

  @parameterized.parameters(
      (sharding.ShardOptions(0, 1), 20),
      (sharding.ShardOptions(0, 2), 10),
      (sharding.ShardOptions(1, 2), 10),
      (sharding.ShardOptions(0, 3), 7),
      (sharding.ShardOptions(1, 3), 7),
      (sharding.ShardOptions(2, 3), 6),
  )
  def test_len(self, shard_options: sharding.ShardOptions, expected_len: int):
    range_ds_for_process = shard_ds.ShardLazyMapDataset(
        self.range_ds,
        shard_options=shard_options,
    )
    self.assertLen(range_ds_for_process, expected_len)

  @parameterized.parameters(itertools.combinations(range(20), 2))
  def test_getitem(self, shard_id: int, num_shards: int):
    shard_options = sharding.ShardOptions(shard_id, num_shards)
    start, stop = sharding.even_split(self.data_len, shard_options)
    expected = self.range_py_list[start:stop]
    ds = shard_ds.ShardLazyMapDataset(
        self.range_ds, shard_options=shard_options
    )
    actual = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(itertools.combinations(range(20), 2))
  def test_iter(self, shard_id: int, num_shards: int):
    shard_options = sharding.ShardOptions(shard_id, num_shards)
    start, stop = sharding.even_split(self.data_len, shard_options)
    expected = self.range_py_list[start:stop]
    ds = shard_ds.ShardLazyMapDataset(
        self.range_ds, shard_options=shard_options
    )
    ds_iter = iter(ds)
    actual = list(ds_iter)
    self.assertSequenceEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
