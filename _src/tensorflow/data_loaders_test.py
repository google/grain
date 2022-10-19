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
"""Unit tests for the data_loaders module."""
from typing import Mapping, Optional
from unittest import mock
from absl.testing import parameterized

from grain._src.core import sharding
from grain._src.core.constants import DATASET_INDEX, RECORD_KEY  # pylint: disable=g-multiple-import
from grain._src.tensorflow import data_loaders
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _dict_to_dataset(d: Mapping[str, np.ndarray]) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(d)


def _dataset_to_dict(ds: tf.data.Dataset) -> Mapping[str, np.ndarray]:
  assert ds.cardinality() > 0 and ds.cardinality() < 1000
  return next(ds.batch(1000).as_numpy_iterator())


class DataLoadersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the data_loaders module."""

  def test_add_global_record_key(self):
    # Let say we have 3 datasets with 3, 8, 5 records each.
    records_per_dataset = (3, 8, 5)
    num_records = sum(records_per_dataset)
    index = {
        RECORD_KEY: np.concatenate((range(3), range(8), range(5)), axis=0),
        DATASET_INDEX: np.asarray(3 * [0] + 8 * [1] + 5 * [2], np.int64),
    }
    index_ds = _dict_to_dataset(index)
    actual_ds = data_loaders._add_global_record_key(
        index_ds,
        records_per_dataset=records_per_dataset,
        output_key="my_output_key")
    actual = _dataset_to_dict(actual_ds)
    expected = index | {"my_output_key": np.arange(num_records, dtype=np.int64)}
    self.assertAllClose(actual, expected)

  @mock.patch.object(tfds.core, "DatasetInfo")
  def test_load_from_tfds_invalid_args(self, tfds_info_mock):
    # Neither name nor tfds_info.
    with self.assertRaisesRegex(ValueError,
                                "Please provide either `name` or `tfds_info`."):
      data_loaders.load_from_tfds(
          split="train", shard_options=sharding.NoSharding())
    # Name and tfds_info.
    with self.assertRaisesRegex(ValueError,
                                "Please provide either `name` or `tfds_info`."):
      data_loaders.load_from_tfds(
          name="my_dataset",
          split="train",
          tfds_info=tfds_info_mock,
          shard_options=sharding.NoSharding())

  @parameterized.parameters([
      (True, 34, None),
      (True, 34, 7),
      (False, 34, 7),
  ])
  def test_load_from_tfds_sampler(self, shuffle: bool, seed: int,
                                  num_epochs: Optional[int]):
    with mock.patch.object(tfds.core, "DatasetInfo") as tfds_info_mock:
      tfds_info_mock.file_format = (
          tfds.core.file_adapters.FileFormat.ARRAY_RECORD)
      loader = data_loaders.load_from_tfds(
          tfds_info=tfds_info_mock,
          split="train",
          shard_options=sharding.NoSharding(),
          seed=seed,
          num_epochs=num_epochs,
          shuffle=shuffle)
      self.assertEqual(loader.sampler.shuffle, shuffle)
      self.assertEqual(loader.sampler.seed, seed)
      self.assertEqual(loader.sampler.num_epochs, num_epochs)


if __name__ == "__main__":
  tf.test.main()
