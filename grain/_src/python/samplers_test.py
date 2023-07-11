# Copyright 2022 Google LLC
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
"""Tests for samplers."""

from absl.testing import absltest
from grain._src.core import sharding
from grain._src.python import record
from grain._src.python import samplers
import numpy as np


class SequentialSamplerTest(absltest.TestCase):

  def test_with_invalid_number_records(self):
    with self.assertRaises(ValueError):
      samplers.SequentialSampler(
          num_records=0, shard_options=sharding.NoSharding()
      )
    with self.assertRaises(ValueError):
      samplers.SequentialSampler(
          num_records=-18, shard_options=sharding.NoSharding()
      )

  def test_sampler_one_shard(self):
    sampler = samplers.SequentialSampler(
        num_records=4, shard_options=sharding.NoSharding()
    )
    actual_record_metadata = list(sampler)
    actual_record_metadata_random_access = [sampler[idx] for idx in range(4)]
    expected_record_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=1, record_key=1),
        record.RecordMetadata(index=2, record_key=2),
        record.RecordMetadata(index=3, record_key=3),
    ]
    self.assertEqual(actual_record_metadata, expected_record_metadata)
    self.assertEqual(
        actual_record_metadata_random_access, expected_record_metadata
    )

  def test_sampler_two_shards(self):
    sharding_option = sharding.ShardOptions(shard_index=0, shard_count=2)
    sampler = samplers.SequentialSampler(
        num_records=4, shard_options=sharding_option
    )
    actual_record_metadata = list(sampler)
    actual_record_metadata_random_access = [
        sampler[idx] for idx in range(0, 4, 2)
    ]
    expected_record_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=2, record_key=2),
    ]
    self.assertEqual(actual_record_metadata, expected_record_metadata)
    self.assertEqual(
        actual_record_metadata_random_access, expected_record_metadata
    )

  def test_sampler_three_shards_with_remainder(self):
    sharding_option = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.SequentialSampler(
        num_records=8, shard_options=sharding_option
    )
    actual_record_metadata = list(sampler)
    actual_record_metadata_random_access = [
        sampler[idx] for idx in range(0, 8, 3)
    ]
    expected_record_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=3),
        record.RecordMetadata(index=6, record_key=6),
    ]
    self.assertEqual(actual_record_metadata, expected_record_metadata)
    self.assertEqual(
        actual_record_metadata_random_access, expected_record_metadata
    )

  def test_sampler_three_shards_no_remainder(self):
    sharding_option = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.SequentialSampler(
        num_records=8, shard_options=sharding_option
    )
    num_records_per_shard = 8 // 3
    total_records = num_records_per_shard * 3
    actual_record_metadata_random_access = [
        sampler[idx] for idx in range(0, total_records, 3)
    ]
    actual_record_metadata = list(sampler)
    expected_record_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=3),
    ]
    self.assertEqual(actual_record_metadata, expected_record_metadata)
    self.assertEqual(
        actual_record_metadata_random_access, expected_record_metadata
    )


class IndexSamplerTest(absltest.TestCase):

  def assertRecordMetadataWithRNGIsEqual(self, actual, expected):
    if len(actual) != len(expected):
      self.fail("Sampler returned incorrect number of record metadata objects")

    for actual_metadata, expected_metadata in zip(actual, expected):
      if (
          actual_metadata.index != expected_metadata.index
          or actual_metadata.record_key != expected_metadata.record_key
      ):
        self.fail("Sampler returned incorrect record metadata")

    rngs = [metadata.rng for metadata in expected]
    if len(rngs) != len(set(rngs)):
      self.fail("RNGs aren't unique for each record metadata object.")

  def test_with_invalid_invalid_num_epochs(self):
    with self.assertRaises(ValueError):
      samplers.IndexSampler(
          num_records=18,
          shard_options=sharding.NoSharding(),
          num_epochs=0,
      )
    with self.assertRaises(ValueError):
      samplers.IndexSampler(
          num_records=18,
          shard_options=sharding.NoSharding(),
          num_epochs=-1,
      )

  def test_simple(self):
    index_sampler = samplers.IndexSampler(
        num_records=200,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=3,
    )
    index_sampler_record_metadata = list(index_sampler)
    self.assertLen(index_sampler_record_metadata, 400)

  def test_invalid_non_integer_seed(self):
    with self.assertRaises(TypeError):
      samplers.IndexSampler(
          num_records=4,
          shard_options=sharding.NoSharding(),
          shuffle=True,
          num_epochs=2,
          seed=(3, 4),  # pytype: disable=wrong-arg-types
      )

  def test_invalid_non_int32_seed(self):
    with self.assertRaises(ValueError):
      samplers.IndexSampler(
          num_records=4,
          shard_options=sharding.NoSharding(),
          shuffle=True,
          num_epochs=2,
          seed=2**32,
      )

  def test_shuffle_no_sharding(self):
    seed = 32
    sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata = list(sampler)
    actual_record_metadata_random_access = [sampler[idx] for idx in range(10)]
    expected_record_metadata = [
        record.RecordMetadata(
            index=0, record_key=2, rng=np.random.Philox(key=seed)
        ),
        record.RecordMetadata(
            index=1, record_key=4, rng=np.random.Philox(key=seed + 1)
        ),
        record.RecordMetadata(
            index=2, record_key=3, rng=np.random.Philox(key=seed + 2)
        ),
        record.RecordMetadata(
            index=3, record_key=0, rng=np.random.Philox(key=seed + 3)
        ),
        record.RecordMetadata(
            index=4, record_key=1, rng=np.random.Philox(key=seed + 4)
        ),
        record.RecordMetadata(
            index=5, record_key=0, rng=np.random.Philox(key=seed + 5)
        ),
        record.RecordMetadata(
            index=6, record_key=2, rng=np.random.Philox(key=seed + 6)
        ),
        record.RecordMetadata(
            index=7, record_key=4, rng=np.random.Philox(key=seed + 7)
        ),
        record.RecordMetadata(
            index=8, record_key=1, rng=np.random.Philox(key=seed + 8)
        ),
        record.RecordMetadata(
            index=9, record_key=3, rng=np.random.Philox(key=seed + 9)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata, expected_record_metadata
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_random_access, expected_record_metadata
    )

  def test_shuffle_and_sharding_drop_remainder_single_epoch(self):
    seed = 32

    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_first_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 6, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(
            index=0, record_key=0, rng=np.random.Philox(key=seed)
        ),
        record.RecordMetadata(
            index=3, record_key=1, rng=np.random.Philox(key=seed + 3)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_second_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 6, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(
            index=1, record_key=2, rng=np.random.Philox(key=seed + 1)
        ),
        record.RecordMetadata(
            index=4, record_key=3, rng=np.random.Philox(key=seed + 4)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_third_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 6, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(
            index=2, record_key=4, rng=np.random.Philox(key=seed + 2)
        ),
        record.RecordMetadata(
            index=5, record_key=5, rng=np.random.Philox(key=seed + 5)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_shuffle_and_sharding_no_drop_remainder_single_epoch(self):
    seed = 32

    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_first_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 8, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(
            index=0, record_key=2, rng=np.random.Philox(key=seed)
        ),
        record.RecordMetadata(
            index=3, record_key=1, rng=np.random.Philox(key=seed + 3)
        ),
        record.RecordMetadata(
            index=6, record_key=0, rng=np.random.Philox(key=seed + 6)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_second_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 8, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(
            index=1, record_key=5, rng=np.random.Philox(key=seed + 1)
        ),
        record.RecordMetadata(
            index=4, record_key=4, rng=np.random.Philox(key=seed + 4)
        ),
        record.RecordMetadata(
            index=7, record_key=3, rng=np.random.Philox(key=seed + 7)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_third_shard,
        shuffle=True,
        num_epochs=1,
        seed=seed,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 8, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(
            index=2, record_key=6, rng=np.random.Philox(key=seed + 2)
        ),
        record.RecordMetadata(
            index=5, record_key=7, rng=np.random.Philox(key=seed + 5)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_shuffle_and_sharding_drop_remainder_multi_epoch(self):
    seed = 32

    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_first_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 6 * 2, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(
            index=0, record_key=0, rng=np.random.Philox(key=seed)
        ),
        record.RecordMetadata(
            index=3, record_key=1, rng=np.random.Philox(key=seed + 3)
        ),
        record.RecordMetadata(
            index=6, record_key=0, rng=np.random.Philox(key=seed + 6)
        ),
        record.RecordMetadata(
            index=9, record_key=1, rng=np.random.Philox(key=seed + 9)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_second_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 6 * 2, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(
            index=1, record_key=2, rng=np.random.Philox(key=seed + 1)
        ),
        record.RecordMetadata(
            index=4, record_key=3, rng=np.random.Philox(key=seed + 4)
        ),
        record.RecordMetadata(
            index=7, record_key=2, rng=np.random.Philox(key=seed + 7)
        ),
        record.RecordMetadata(
            index=10, record_key=3, rng=np.random.Philox(key=seed + 10)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_third_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 6 * 2, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(
            index=2, record_key=4, rng=np.random.Philox(key=seed + 2)
        ),
        record.RecordMetadata(
            index=5, record_key=5, rng=np.random.Philox(key=seed + 5)
        ),
        record.RecordMetadata(
            index=8, record_key=4, rng=np.random.Philox(key=seed + 8)
        ),
        record.RecordMetadata(
            index=11, record_key=5, rng=np.random.Philox(key=seed + 11)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_shuffle_and_sharding_no_drop_remainder_multi_epoch(self):
    seed = 32

    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_first_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 8 * 2, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(
            index=0, record_key=2, rng=np.random.Philox(key=seed)
        ),
        record.RecordMetadata(
            index=3, record_key=1, rng=np.random.Philox(key=seed + 3)
        ),
        record.RecordMetadata(
            index=6, record_key=0, rng=np.random.Philox(key=seed + 6)
        ),
        record.RecordMetadata(
            index=9, record_key=0, rng=np.random.Philox(key=seed + 9)
        ),
        record.RecordMetadata(
            index=12, record_key=2, rng=np.random.Philox(key=seed + 12)
        ),
        record.RecordMetadata(
            index=15, record_key=1, rng=np.random.Philox(key=seed + 15)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_second_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 8 * 2, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(
            index=1, record_key=5, rng=np.random.Philox(key=seed + 1)
        ),
        record.RecordMetadata(
            index=4, record_key=4, rng=np.random.Philox(key=seed + 4)
        ),
        record.RecordMetadata(
            index=7, record_key=3, rng=np.random.Philox(key=seed + 7)
        ),
        record.RecordMetadata(
            index=10, record_key=3, rng=np.random.Philox(key=seed + 10)
        ),
        record.RecordMetadata(
            index=13, record_key=5, rng=np.random.Philox(key=seed + 13)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_third_shard,
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 8 * 2, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(
            index=2, record_key=6, rng=np.random.Philox(key=seed + 2)
        ),
        record.RecordMetadata(
            index=5, record_key=7, rng=np.random.Philox(key=seed + 5)
        ),
        record.RecordMetadata(
            index=8, record_key=6, rng=np.random.Philox(key=seed + 8)
        ),
        record.RecordMetadata(
            index=11, record_key=7, rng=np.random.Philox(key=seed + 11)
        ),
        record.RecordMetadata(
            index=14, record_key=6, rng=np.random.Philox(key=seed + 14)
        ),
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertRecordMetadataWithRNGIsEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_sharding_no_shuffle_drop_remainder_single_epoch(self):
    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_first_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 6, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(index=0, record_key=0, rng=None),
        record.RecordMetadata(index=3, record_key=1, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_second_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 6, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(index=1, record_key=2, rng=None),
        record.RecordMetadata(index=4, record_key=3, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_third_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 6, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(index=2, record_key=4, rng=None),
        record.RecordMetadata(index=5, record_key=5, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_sharding_no_shuffle_no_drop_remainder_single_epoch(self):
    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_first_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 8, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(index=0, record_key=0, rng=None),
        record.RecordMetadata(index=3, record_key=1, rng=None),
        record.RecordMetadata(index=6, record_key=2, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_second_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 8, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(index=1, record_key=3, rng=None),
        record.RecordMetadata(index=4, record_key=4, rng=None),
        record.RecordMetadata(index=7, record_key=5, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_third_shard,
        shuffle=False,
        num_epochs=1,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 8, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(index=2, record_key=6, rng=None),
        record.RecordMetadata(index=5, record_key=7, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_sharding_no_shuffle_drop_remainder_multi_epoch(self):
    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_first_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 6 * 2, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(index=0, record_key=0, rng=None),
        record.RecordMetadata(index=3, record_key=1, rng=None),
        record.RecordMetadata(index=6, record_key=0, rng=None),
        record.RecordMetadata(index=9, record_key=1, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_second_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 6 * 2, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(index=1, record_key=2, rng=None),
        record.RecordMetadata(index=4, record_key=3, rng=None),
        record.RecordMetadata(index=7, record_key=2, rng=None),
        record.RecordMetadata(index=10, record_key=3, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=6,
        shard_options=sharding_option_third_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 6 * 2, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(index=2, record_key=4, rng=None),
        record.RecordMetadata(index=5, record_key=5, rng=None),
        record.RecordMetadata(index=8, record_key=4, rng=None),
        record.RecordMetadata(index=11, record_key=5, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_sharding_no_shuffle_no_drop_remainder_multi_epoch(self):
    sharding_option_first_shard = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler_first_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_first_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_first_shard = list(sampler_first_shard)
    actual_record_metadata_first_shard_random_access = [
        sampler_first_shard[idx] for idx in range(0, 8 * 2, 3)
    ]
    expected_record_metadata_first_shard = [
        record.RecordMetadata(index=0, record_key=0, rng=None),
        record.RecordMetadata(index=3, record_key=1, rng=None),
        record.RecordMetadata(index=6, record_key=2, rng=None),
        record.RecordMetadata(index=9, record_key=0, rng=None),
        record.RecordMetadata(index=12, record_key=1, rng=None),
        record.RecordMetadata(index=15, record_key=2, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_first_shard, expected_record_metadata_first_shard
    )
    self.assertEqual(
        actual_record_metadata_first_shard_random_access,
        expected_record_metadata_first_shard,
    )

    sharding_option_second_shard = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler_second_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_second_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_second_shard = list(sampler_second_shard)
    actual_record_metadata_second_shard_random_access = [
        sampler_second_shard[idx] for idx in range(1, 8 * 2, 3)
    ]
    expected_record_metadata_second_shard = [
        record.RecordMetadata(index=1, record_key=3, rng=None),
        record.RecordMetadata(index=4, record_key=4, rng=None),
        record.RecordMetadata(index=7, record_key=5, rng=None),
        record.RecordMetadata(index=10, record_key=3, rng=None),
        record.RecordMetadata(index=13, record_key=4, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_second_shard,
        expected_record_metadata_second_shard,
    )
    self.assertEqual(
        actual_record_metadata_second_shard_random_access,
        expected_record_metadata_second_shard,
    )

    sharding_option_third_shard = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler_third_shard = samplers.IndexSampler(
        num_records=8,
        shard_options=sharding_option_third_shard,
        shuffle=False,
        num_epochs=2,
    )
    actual_record_metadata_third_shard = list(sampler_third_shard)
    actual_record_metadata_third_shard_random_access = [
        sampler_third_shard[idx] for idx in range(2, 8 * 2, 3)
    ]
    expected_record_metadata_third_shard = [
        record.RecordMetadata(index=2, record_key=6, rng=None),
        record.RecordMetadata(index=5, record_key=7, rng=None),
        record.RecordMetadata(index=8, record_key=6, rng=None),
        record.RecordMetadata(index=11, record_key=7, rng=None),
        record.RecordMetadata(index=14, record_key=6, rng=None),
    ]
    self.assertEqual(
        actual_record_metadata_third_shard, expected_record_metadata_third_shard
    )
    self.assertEqual(
        actual_record_metadata_third_shard_random_access,
        expected_record_metadata_third_shard,
    )

  def test_determinism(self):
    seed = 32
    first_sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    first_sampler_actual_record_metadata = list(first_sampler)
    first_sampler_actual_record_metadata_random_access = [
        first_sampler[idx] for idx in range(5)
    ]
    second_sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=seed,
    )
    second_sampler_actual_record_metadata = list(second_sampler)
    second_sampler_actual_record_metadata_random_access = [
        second_sampler[idx] for idx in range(5)
    ]
    self.assertRecordMetadataWithRNGIsEqual(
        first_sampler_actual_record_metadata,
        second_sampler_actual_record_metadata,
    )
    self.assertRecordMetadataWithRNGIsEqual(
        first_sampler_actual_record_metadata_random_access,
        second_sampler_actual_record_metadata_random_access,
    )


if __name__ == "__main__":
  absltest.main()
