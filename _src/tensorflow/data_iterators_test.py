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
"""Unit tests for the data_iterators module."""
import dataclasses
import tempfile
from typing import Optional
from unittest import mock

from absl.testing import parameterized
import chex
from clu.data import dataset_iterator
from etils import epath
from grain._src.core.constants import INDEX  # pylint: disable=g-multiple-import
from grain._src.tensorflow import data_iterators
from grain._src.tensorflow import index_dataset
import numpy as np
import tensorflow as tf

ArraySpec = dataset_iterator.ArraySpec


@dataclasses.dataclass(frozen=True)
class DummySampler:
  """Dummy IndexSampler that samples range(0, 10)."""

  shard_index: int = 0
  shard_count: int = 1

  def as_dict(self):
    return dataclasses.asdict(self)

  def as_index_dataset(self,
                       start_index: index_dataset.Index) -> tf.data.Dataset:
    if isinstance(start_index, index_dataset.FirstIndex):
      start_index = self.shard_index
    elif isinstance(start_index, index_dataset.NextIndex):
      start_index = start_index.last_seen_index + self.shard_count
    assert isinstance(start_index, int)
    return tf.data.Dataset.range(start_index, 10)


@dataclasses.dataclass(frozen=True)
class DummyDataLoader:

  source: str = "DummySource"
  sampler: index_dataset.TfIndexSampler = DummySampler()
  batch_size: Optional[int] = None

  def as_dataset(self, start_index: index_dataset.Index) -> tf.data.Dataset:
    ds = self.sampler.as_index_dataset(start_index)
    ds = ds.map(lambda x: {INDEX: x, "number": tf.cast(2 * x + 1, tf.uint32)})
    if self.batch_size:
      ds = ds.batch(self.batch_size, drop_remainder=True)
    return ds


class DataIteratorsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the data_iterators module."""

  def test_next(self):
    it = data_iterators.TfGrainDatasetIterator(DummyDataLoader())
    self.assertAllEqual(it.element_spec, {
        INDEX: ArraySpec(np.int64, ()),
        "number": ArraySpec(np.uint32, ())
    })
    for i in range(10):
      self.assertAllEqual(next(it), {INDEX: i, "number": 2*i+1})
    self.assertRaises(StopIteration, next, it)  # End of iterator.
    self.assertRaises(StopIteration, next, it)  # Iterator stays invalid
    it.reset()
    self.assertAllEqual(next(it), {INDEX: 0, "number": 1})

  def test_next_with_drop_grain_meta_features(self):
    options = data_iterators.IteratorOptions(drop_grain_meta_features=True)
    it = data_iterators.TfGrainDatasetIterator(DummyDataLoader(), options)
    self.assertAllEqual(it.element_spec, {"number": ArraySpec(np.uint32, ())})
    self.assertAllEqual(next(it), {"number": 1})
    self.assertAllEqual(next(it), {"number": 3})
    self.assertAllEqual(next(it), {"number": 5})

  @parameterized.parameters(1, 2)
  def test_next_reshape_for_local_devices(self, num_devices: int):
    batch_size = 4
    with mock.patch.object(
        data_iterators.jax, "local_device_count", return_value=num_devices):
      options = data_iterators.IteratorOptions(reshape_for_local_devices=True)
      it = data_iterators.TfGrainDatasetIterator(
          DummyDataLoader(batch_size=batch_size), options)
      expected_shape = (num_devices, batch_size // num_devices)
      self.assertAllEqual(
          it.element_spec, {
              INDEX: ArraySpec(np.int64, expected_shape),
              "number": ArraySpec(np.uint32, expected_shape),
          })
      if num_devices == 1:
        chex.assert_trees_all_close(
            next(it), {
                INDEX: np.asarray([[0, 1, 2, 3]], np.int64),
                "number": np.asarray([[1, 3, 5, 7]], np.uint32),
            })
        chex.assert_trees_all_close(
            next(it), {
                INDEX: np.asarray([[4, 5, 6, 7]], np.int64),
                "number": np.asarray([[9, 11, 13, 15]], np.uint32),
            })
      else:
        assert num_devices == 2
        chex.assert_trees_all_close(
            next(it), {
                INDEX: np.asarray([[0, 1], [2, 3]], np.int64),
                "number": np.asarray([[1, 3], [5, 7]], np.uint32),
            })
        chex.assert_trees_all_close(
            next(it), {
                INDEX: np.asarray([[4, 5], [6, 7]], np.int64),
                "number": np.asarray([[9, 11], [13, 15]], np.uint32),
            })

  def test_save(self):
    testdir = epath.Path(tempfile.mkdtemp()) / "test"
    testdir.mkdir(parents=True, exist_ok=False)
    it = data_iterators.TfGrainDatasetIterator(DummyDataLoader())
    checkpoint0 = testdir / "step0.json"
    it.save(checkpoint0)
    self.assertEqual(
        checkpoint0.read_text(), """{
    "last_seen_index": null,
    "source": "'DummySource'",
    "sampler": {
        "shard_index": 0,
        "shard_count": 1
    }
}""")
    checkpoint2 = testdir / "step2.json"
    next(it)
    next(it)
    it.save(checkpoint2)
    self.assertEqual(
        checkpoint2.read_text(), """{
    "last_seen_index": 1,
    "source": "'DummySource'",
    "sampler": {
        "shard_index": 0,
        "shard_count": 1
    }
}""")

  def test_restore(self):
    testdir = epath.Path(tempfile.mkdtemp()) / "test"
    testdir.mkdir(parents=True, exist_ok=False)
    it = data_iterators.TfGrainDatasetIterator(DummyDataLoader())
    checkpoint3 = testdir / "step3.json"
    checkpoint3.write_text("""{
    "last_seen_index": 2,
    "source": "'DummySource'",
    "sampler": {
        "shard_index": 0,
        "shard_count": 1
    }
}""")
    it.restore(checkpoint3)
    self.assertAllEqual(next(it), {INDEX: 3, "number": 7})

  def test_restore_fails_for_different_shard_options(self):
    testdir = epath.Path(tempfile.mkdtemp()) / "test"
    testdir.mkdir(parents=True, exist_ok=False)
    it = data_iterators.TfGrainDatasetIterator(DummyDataLoader())
    checkpoint3 = testdir / "step3.json"
    checkpoint3.write_text("""{
    "last_seen_index": 2,
    "source": "'DummySource'",
    "sampler": {
        "shard_index": 0,
        "shard_count": 2
    }
}""")
    with self.assertRaisesRegex(
        ValueError,
        "Sampler specification in checkpoint doesn't match expected"):
      it.restore(checkpoint3)


if __name__ == "__main__":
  tf.test.main()
