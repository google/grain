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
"""Tests for samplers."""
from collections.abc import Sequence

from absl.testing import absltest
from grain._src.core import sharding
from grain._src.python import record
from grain._src.python import samplers

from absl.testing import parameterized


def _get_all_metadata(
    sampler: samplers.Sampler, shard_options: sharding.ShardOptions
) -> Sequence[record.RecordMetadata]:
  metadata = []
  i = shard_options.shard_index
  while True:
    try:
      metadata.append(sampler[i])
    except IndexError:
      break
    i += shard_options.shard_count
  return metadata


def _remove_rngs(
    metadata: Sequence[record.RecordMetadata],
) -> Sequence[record.RecordMetadata]:
  return [
      record.RecordMetadata(index=m.index, record_key=m.record_key)
      for m in metadata
  ]


class SequentialSamplerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'negative_index', 'num_records': 42, 'index': -1},
      {'testcase_name': 'too_large_index', 'num_records': 42, 'index': 42},
  )
  def test_index_out_of_bounds_raises_index_error(
      self, num_records: int, index: int
  ):
    sampler = samplers.SequentialSampler(
        num_records=num_records, shard_options=sharding.NoSharding()
    )
    with self.assertRaises(IndexError):
      _ = sampler[index]

  def test_with_invalid_number_records(self):
    with self.assertRaises(ValueError):
      samplers.SequentialSampler(
          num_records=0, shard_options=sharding.NoSharding()
      )
    with self.assertRaises(ValueError):
      samplers.SequentialSampler(
          num_records=-18, shard_options=sharding.NoSharding()
      )

  def test_no_sharding(self):
    sampler = samplers.SequentialSampler(
        num_records=4, shard_options=sharding.NoSharding()
    )
    actual_metadata = _get_all_metadata(
        sampler, shard_options=sharding.NoSharding()
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=1, record_key=1),
        record.RecordMetadata(index=2, record_key=2),
        record.RecordMetadata(index=3, record_key=3),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sampler_sharding(self):
    shard_options = sharding.ShardOptions(shard_index=0, shard_count=2)
    sampler = samplers.SequentialSampler(
        num_records=4, shard_options=shard_options
    )
    actual_metadata = _get_all_metadata(sampler, shard_options=shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=2, record_key=2),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sampler_sharding_no_drop_remainder(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.SequentialSampler(
        num_records=8, shard_options=shard_options
    )
    actual_metadata = _get_all_metadata(sampler, shard_options=shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=3),
        record.RecordMetadata(index=6, record_key=6),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sampler_sharding_drop_remainder(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.SequentialSampler(
        num_records=8, shard_options=shard_options
    )
    actual_metadata = _get_all_metadata(sampler, shard_options=shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=3),
    ]
    self.assertEqual(actual_metadata, expected_metadata)


class IndexSamplerTest(absltest.TestCase):

  def assertRngsAreUnique(self, actual: Sequence[record.RecordMetadata]):
    actual_floats = [metadata.rng.random() for metadata in actual]
    if len(actual_floats) != len(set(actual_floats)):
      self.fail(
          'At least 2 RNGs returned the same random number. Metadata with'
          f' RNGs: {actual}'
      )

  def assertRecordMetadata(
      self,
      sampler: samplers.Sampler,
      shard_options: sharding.ShardOptions,
      expected_metadata: Sequence[record.RecordMetadata],
  ):
    actual_metadata = _get_all_metadata(sampler, shard_options)
    self.assertRngsAreUnique(actual_metadata)
    self.assertEqual(_remove_rngs(actual_metadata), expected_metadata)

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
    actual_metadata = _get_all_metadata(index_sampler, sharding.NoSharding())
    self.assertLen(actual_metadata, 400)

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

  def test_negative_index_raises_index_error(self):
    sampler = samplers.IndexSampler(
        num_records=42, shard_options=sharding.NoSharding()
    )
    with self.assertRaises(IndexError):
      sampler[-1]  # pylint: disable=pointless-statement

  def test_index_too_large_raises_index_error(self):
    sampler = samplers.IndexSampler(
        num_records=42, shard_options=sharding.NoSharding(), num_epochs=1
    )
    with self.assertRaises(IndexError):
      sampler[42]  # pylint: disable=pointless-statement

  def test_shuffle_no_sharding(self):
    sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=2),
        record.RecordMetadata(index=1, record_key=4),
        record.RecordMetadata(index=2, record_key=3),
        record.RecordMetadata(index=3, record_key=0),
        record.RecordMetadata(index=4, record_key=1),
        record.RecordMetadata(index=5, record_key=0),
        record.RecordMetadata(index=6, record_key=2),
        record.RecordMetadata(index=7, record_key=4),
        record.RecordMetadata(index=8, record_key=1),
        record.RecordMetadata(index=9, record_key=3),
    ]
    self.assertRecordMetadata(sampler, sharding.NoSharding(), expected_metadata)

  def test_shuffle_and_sharding_drop_remainder_single_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=2),
        record.RecordMetadata(index=4, record_key=3),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=4),
        record.RecordMetadata(index=5, record_key=5),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

  def test_shuffle_and_sharding_no_drop_remainder_single_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=2),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=0),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=5),
        record.RecordMetadata(index=4, record_key=4),
        record.RecordMetadata(index=7, record_key=3),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=1,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=6),
        record.RecordMetadata(index=5, record_key=7),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

  def test_shuffle_and_sharding_drop_remainder_multi_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=0),
        record.RecordMetadata(index=9, record_key=1),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=2),
        record.RecordMetadata(index=4, record_key=3),
        record.RecordMetadata(index=7, record_key=2),
        record.RecordMetadata(index=10, record_key=3),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=4),
        record.RecordMetadata(index=5, record_key=5),
        record.RecordMetadata(index=8, record_key=4),
        record.RecordMetadata(index=11, record_key=5),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

  def test_shuffle_and_sharding_no_drop_remainder_multi_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=2),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=0),
        record.RecordMetadata(index=9, record_key=0),
        record.RecordMetadata(index=12, record_key=2),
        record.RecordMetadata(index=15, record_key=1),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=5),
        record.RecordMetadata(index=4, record_key=4),
        record.RecordMetadata(index=7, record_key=3),
        record.RecordMetadata(index=10, record_key=3),
        record.RecordMetadata(index=13, record_key=5),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=6),
        record.RecordMetadata(index=5, record_key=7),
        record.RecordMetadata(index=8, record_key=6),
        record.RecordMetadata(index=11, record_key=7),
        record.RecordMetadata(index=14, record_key=6),
    ]
    self.assertRecordMetadata(sampler, shard_options, expected_metadata)

  def test_sharding_no_shuffle_drop_remainder_single_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=2),
        record.RecordMetadata(index=4, record_key=3),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=4),
        record.RecordMetadata(index=5, record_key=5),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sharding_no_shuffle_no_drop_remainder_single_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=2),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=3),
        record.RecordMetadata(index=4, record_key=4),
        record.RecordMetadata(index=7, record_key=5),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=1,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=6),
        record.RecordMetadata(index=5, record_key=7),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sharding_no_shuffle_drop_remainder_multi_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=0),
        record.RecordMetadata(index=9, record_key=1),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=2),
        record.RecordMetadata(index=4, record_key=3),
        record.RecordMetadata(index=7, record_key=2),
        record.RecordMetadata(index=10, record_key=3),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=True
    )
    sampler = samplers.IndexSampler(
        num_records=6,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=4),
        record.RecordMetadata(index=5, record_key=5),
        record.RecordMetadata(index=8, record_key=4),
        record.RecordMetadata(index=11, record_key=5),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_sharding_no_shuffle_no_drop_remainder_multi_epoch(self):
    shard_options = sharding.ShardOptions(
        shard_index=0, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=0, record_key=0),
        record.RecordMetadata(index=3, record_key=1),
        record.RecordMetadata(index=6, record_key=2),
        record.RecordMetadata(index=9, record_key=0),
        record.RecordMetadata(index=12, record_key=1),
        record.RecordMetadata(index=15, record_key=2),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=1, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=1, record_key=3),
        record.RecordMetadata(index=4, record_key=4),
        record.RecordMetadata(index=7, record_key=5),
        record.RecordMetadata(index=10, record_key=3),
        record.RecordMetadata(index=13, record_key=4),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

    shard_options = sharding.ShardOptions(
        shard_index=2, shard_count=3, drop_remainder=False
    )
    sampler = samplers.IndexSampler(
        num_records=8,
        shard_options=shard_options,
        shuffle=False,
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(sampler, shard_options)
    expected_metadata = [
        record.RecordMetadata(index=2, record_key=6),
        record.RecordMetadata(index=5, record_key=7),
        record.RecordMetadata(index=8, record_key=6),
        record.RecordMetadata(index=11, record_key=7),
        record.RecordMetadata(index=14, record_key=6),
    ]
    self.assertEqual(actual_metadata, expected_metadata)

  def test_determinism(self):
    first_sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    first_metadata = _get_all_metadata(first_sampler, sharding.NoSharding())
    second_sampler = samplers.IndexSampler(
        num_records=5,
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=2,
        seed=32,
    )
    second_metadata = _get_all_metadata(second_sampler, sharding.NoSharding())
    self.assertEqual(len(first_metadata), len(second_metadata))
    for m1, m2 in zip(first_metadata, second_metadata):
      if m1.rng.random() != m2.rng.random():
        self.fail('Metadata RNGs returned different floats.')
    self.assertEqual(
        _remove_rngs(first_metadata), _remove_rngs(second_metadata)
    )


if __name__ == '__main__':
  absltest.main()
