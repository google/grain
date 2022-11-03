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

  def random_map(self, features, seed):
    features["y"] = tf.random.stateless_uniform([],
                                                seed,
                                                dtype=tf.int64,
                                                maxval=100)
    return features


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

  def _create_dataset(self, with_seed: bool = False) -> tf.data.Dataset:
    elements = []
    for index in range(4):
      element = {constants.INDEX: tf.cast(index, tf.int64), "x": index}
      if with_seed:
        element[constants.SEED] = (42, index)
      elements.append(element)
    return tf.data.experimental.from_list(elements)

  def test_map_transform(self):
    ds = self._create_dataset()
    ds = transforms.apply_transformations(ds, [Square()])
    ds = [element["x"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [0, 1, 4, 9])

  def test_clu_preprocess_op(self):
    ds = self._create_dataset()
    ds = transforms.apply_transformations(ds, [SquareAsPreprocessOp()])
    ds = [element["x"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [0, 1, 4, 9])

  def test_map_over_dataset_strict(self):
    ds = self._create_dataset()
    with self.assertRaisesRegex(ValueError, "Could not apply transform"):
      transforms.apply_transformations(ds, [square_in_seqio], strict=True)

  def test_map_over_dataset(self):
    ds = self._create_dataset()
    ds = transforms.apply_transformations(ds, [square_in_seqio], strict=False)
    ds = [element["x"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [0, 1, 4, 9])

  def test_random_map_transform(self):
    ds = self._create_dataset(with_seed=True)
    ds = transforms.apply_transformations(ds, [AddRandomNumber()])
    ds = [element["y"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [45, 58, 69, 6])

  def test_filter_transform(self):
    ds = self._create_dataset()
    ds = transforms.apply_transformations(ds, [FilterOdd()])
    ds = [element["x"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [0, 2])

  def test_cache(self):
    ds = self._create_dataset()
    ds = transforms.apply_transformations(ds, [transforms.CacheTransform()])
    ds = [element["x"] for element in ds.as_numpy_iterator()]
    self.assertAllEqual(ds, [0, 1, 2, 3])


if __name__ == "__main__":
  tf.test.main()
