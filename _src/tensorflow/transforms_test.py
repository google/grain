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
"""Unit tests for the transforms module."""
from absl.testing import parameterized
from grain._src.core import constants
from grain._src.tensorflow import transforms
import seqio
import tensorflow as tf


class Square(transforms.MapTransform):

  def map(self, features):
    features["x"] = features["x"]**2
    return features


class AddRandomNumber(transforms.RandomMapTransform):

  def random_map(self, x, seed):
    return {
        "y": tf.random.stateless_uniform([], seed, dtype=tf.int64, maxval=100)
    }


class FilterOdd(transforms.FilterTransform):

  def filter(self, features):
    return features["x"] % 2 == 0


class SquareAsPreprocessOp:

  def __call__(self, features):
    features["x"] = features["x"]**2
    return features


@seqio.map_over_dataset
def square_in_seqio(features):
  features["x"] = features["x"]**2
  return features


class TransformsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the transforms module."""

  @parameterized.parameters(Square(), SquareAsPreprocessOp())
  def test_map(self, transform):
    ds = tf.data.Dataset.range(4).map(lambda x: {"x": x})
    ds = transforms.apply_transformations(ds, [transform])
    ds = list(ds.as_numpy_iterator())
    self.assertAllEqual(ds, [{"x": 0}, {"x": 1}, {"x": 4}, {"x": 9}])

  def test_map_seqio_preprocessor(self):
    ds = tf.data.Dataset.range(4).map(lambda x: {"x": x})
    with self.assertRaisesRegex(ValueError, "Using unsafe transformation"):
      transforms.apply_transformations(ds, [square_in_seqio], strict=True)

  def test_map_seqio_preprocessor_strict(self):
    ds = tf.data.Dataset.range(4).map(lambda x: {"x": x})
    ds = transforms.apply_transformations(ds, [square_in_seqio], strict=False)
    ds = list(ds.as_numpy_iterator())
    self.assertAllEqual(ds, [{"x": 0}, {"x": 1}, {"x": 4}, {"x": 9}])

  def test_random_map_transform(self):
    ds = tf.data.experimental.from_list([{
        constants.SEED: (1, 2)
    }, {
        constants.SEED: (3, 4)
    }])
    ds = transforms.apply_transformations(ds, [AddRandomNumber()])
    ds = [e["y"] for e in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [81, 35])

  def test_filter_transform(self):
    ds = tf.data.Dataset.range(4).map(lambda x: {"x": x})
    ds = transforms.apply_transformations(ds, [FilterOdd()])
    ds = list(ds.as_numpy_iterator())
    self.assertAllEqual(ds, [{"x": 0}, {"x": 2}])

  def test_cache(self):
    ds = tf.data.Dataset.range(4)
    ds = transforms.apply_transformations(ds, [transforms.CacheTransform()])
    ds = list(ds.as_numpy_iterator())
    self.assertAllEqual(ds, [0, 1, 2, 3])


if __name__ == "__main__":
  tf.test.main()
