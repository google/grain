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
import dataclasses
from typing import Mapping, Optional
from unittest import mock

from absl.testing import parameterized
from grain._src.core import constants
from grain._src.core import sharding
from grain._src.tensorflow import batching
from grain._src.tensorflow import data_loaders
from grain._src.tensorflow import data_sources
from grain._src.tensorflow import index_dataset
from grain._src.tensorflow import transforms
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _dict_to_dataset(d: Mapping[str, np.ndarray]) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(d)


def _dataset_to_dict(ds: tf.data.Dataset) -> Mapping[str, np.ndarray]:
  assert ds.cardinality() > 0 and ds.cardinality() < 1000
  return next(ds.batch(1000).as_numpy_iterator())


@dataclasses.dataclass(frozen=True)
class _DummyParseFn:
  """Dummy ParseFn for TfDataSource."""

  def __call__(self, record: tf.Tensor):
    return {"value": tf.fill([2, 3], record)}


class _FilterEvenIndices(transforms.FilterTransform):

  def filter(self, features):
    return features[constants.INDEX] % 2 != 0


@dataclasses.dataclass(frozen=True)
class _AddRandomValue(transforms.RandomMapTransform):
  feature_name: str
  minval: float
  maxval: float

  def random_map(self, features, seed):
    features[self.feature_name] = tf.random.stateless_uniform(
        [], seed, minval=self.minval, maxval=self.maxval
    )
    return features


class DataLoadersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the data_loaders module."""

  def test_add_global_record_key(self):
    # Let say we have 3 datasets with 3, 8, 5 records each.
    records_per_dataset = (3, 8, 5)
    num_records = sum(records_per_dataset)
    index = {
        constants.RECORD_KEY: np.concatenate(
            (range(3), range(8), range(5)), axis=0
        ),
        constants.DATASET_INDEX: np.asarray(
            3 * [0] + 8 * [1] + 5 * [2], np.int64
        ),
    }
    index_ds = _dict_to_dataset(index)
    actual_ds = data_loaders._add_global_record_key(
        index_ds,
        records_per_dataset=records_per_dataset,
        output_key="my_output_key",
    )
    actual = _dataset_to_dict(actual_ds)
    expected = index | {"my_output_key": np.arange(num_records, dtype=np.int64)}
    self.assertAllClose(actual, expected)

  @mock.patch.object(tfds.core, "DatasetInfo")
  def test_load_from_tfds_invalid_args(self, tfds_info_mock):
    # Neither name nor tfds_info.
    with self.assertRaisesRegex(
        ValueError, "Please provide either `name` or `tfds_info`."
    ):
      data_loaders.load_from_tfds(
          split="train", shard_options=sharding.NoSharding()
      )
    # Name and tfds_info.
    with self.assertRaisesRegex(
        ValueError, "Please provide either `name` or `tfds_info`."
    ):
      data_loaders.load_from_tfds(
          name="my_dataset",
          split="train",
          tfds_info=tfds_info_mock,
          shard_options=sharding.NoSharding(),
      )

  @parameterized.parameters([
      (True, 34, None),
      (True, 34, 7),
      (False, 34, 7),
  ])
  def test_load_from_tfds_sampler(
      self, shuffle: bool, seed: int, num_epochs: Optional[int]
  ):
    with mock.patch.object(tfds.core, "DatasetInfo") as tfds_info_mock:
      tfds_info_mock.file_format = (
          tfds.core.file_adapters.FileFormat.ARRAY_RECORD
      )
      loader = data_loaders.load_from_tfds(
          tfds_info=tfds_info_mock,
          split="train",
          shard_options=sharding.NoSharding(),
          seed=seed,
          num_epochs=num_epochs,
          shuffle=shuffle,
      )
      self.assertEqual(loader.sampler.shuffle, shuffle)
      self.assertEqual(loader.sampler.seed, seed)
      self.assertEqual(loader.sampler.num_epochs, num_epochs)

  def test_mixture_data_loader(self):
    values1 = np.random.random((10,))
    values2 = np.random.random((15,))
    sources = [
        data_sources.TfInMemoryDataSource(values1, _DummyParseFn()),
        data_sources.TfInMemoryDataSource(values2, _DummyParseFn()),
    ]
    sampler = index_dataset.TfMixtureIndexSampler(
        [len(s) for s in sources],
        shard_options=sharding.NoSharding(),
        shuffle=False,
    )
    loader = data_loaders.TfMixtureDataLoader(
        sources=sources,
        transformations_per_source=2 * [[]],
        sampler=sampler,
        transformations=[batching.TfBatch(4, drop_remainder=False)],
    )
    self.assertLen(loader.source, 25)
    it = iter(loader)
    batch = next(it)
    self.assertAllEqual(batch[constants.INDEX], [0, 1, 2, 3])
    self.assertAllEqual(batch[constants.DATASET_INDEX], [0, 1, 0, 1])

  def test_mixture_data_loader_with_random_map(self):
    values1 = np.random.random((10,))
    values2 = np.random.random((15,))
    sources = [
        data_sources.TfInMemoryDataSource(values1, _DummyParseFn()),
        data_sources.TfInMemoryDataSource(values2, _DummyParseFn()),
    ]
    sampler = index_dataset.TfMixtureIndexSampler(
        [len(s) for s in sources],
        shard_options=sharding.NoSharding(),
        shuffle=False,
        seed=32,
    )
    loader = data_loaders.TfMixtureDataLoader(
        sources=sources,
        transformations_per_source=[
            [_AddRandomValue("x", 0, 1), _AddRandomValue("y", 0, 1)],
            [_AddRandomValue("x", 1, 100), _AddRandomValue("y", 1, 100)],
        ],
        sampler=sampler,
        transformations=[batching.TfBatch(4, drop_remainder=False)],
    )
    self.assertLen(loader.source, 25)
    it = iter(loader)
    batch = next(it)
    self.assertAllEqual(batch[constants.INDEX], [0, 1, 2, 3])
    self.assertAllEqual(batch[constants.DATASET_INDEX], [0, 1, 0, 1])
    for i in range(4):
      x = batch["x"][i]
      if batch[constants.DATASET_INDEX][i] == 1:
        self.assertBetween(x, 1, 100)
      else:
        self.assertBetween(x, 0, 1)

  def test_mixture_data_loader_with_filter(self):
    values1 = np.random.random((10,))
    values2 = np.random.random((15,))
    sources = [
        data_sources.TfInMemoryDataSource(values1, _DummyParseFn()),
        data_sources.TfInMemoryDataSource(values2, _DummyParseFn()),
    ]
    sampler = index_dataset.TfMixtureIndexSampler(
        [len(s) for s in sources],
        proportions=[0.6, 0.3],
        shard_options=sharding.NoSharding(),
        shuffle=False,
    )
    loader = data_loaders.TfMixtureDataLoader(
        sources=sources,
        transformations_per_source=[[_FilterEvenIndices()], []],
        sampler=sampler,
        transformations=[batching.TfBatch(8, drop_remainder=False)],
    )
    self.assertLen(loader.source, 25)
    it = iter(loader)
    batch = next(it)
    self.assertAllEqual(
        batch[constants.DATASET_INDEX], [0, 1, 0, 1, 0, 1, 0, 1]
    )
    self.assertAllEqual(batch[constants.INDEX], [1, 2, 3, 5, 7, 8, 9, 11])
    self.assertAllEqual(batch["value"].shape, (8, 2, 3))


if __name__ == "__main__":
  tf.test.main()
