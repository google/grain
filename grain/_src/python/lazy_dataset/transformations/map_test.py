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
"""Tests for map transformation."""

import dataclasses

from absl.testing import absltest
from grain._src.core import transforms
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations.map import MapLazyIterDataset
from grain._src.python.lazy_dataset.transformations.map import MapLazyMapDataset
import numpy as np


@dataclasses.dataclass(frozen=True)
class MapWithNoTransform(transforms.MapTransform):

  def map(self, element: int):
    return element


@dataclasses.dataclass(frozen=True)
class MapWithTransform(transforms.MapTransform):

  def map(self, element: int):
    return element + 1


@dataclasses.dataclass(frozen=True)
class RandomMapWithTransform(transforms.RandomMapTransform):

  def random_map(self, element: int, rng: np.random.Generator):
    delta = 0.1
    return element + rng.uniform(-delta, delta)


class MapLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = lazy_dataset.RangeLazyMapDataset(0, 10)

  def test_map_size(self):
    map_ds_no_transform = MapLazyMapDataset(self.range_ds, MapWithNoTransform())
    map_ds_with_transform = MapLazyMapDataset(self.range_ds, MapWithTransform())
    map_ds_with_random_transform = MapLazyMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    self.assertLen(map_ds_no_transform, len(self.range_ds))
    self.assertLen(map_ds_with_transform, len(self.range_ds))
    self.assertLen(map_ds_with_random_transform, len(self.range_ds))

  def test_map_data_no_transform(self):
    map_ds_no_transform = MapLazyMapDataset(self.range_ds, MapWithNoTransform())
    expected_data = [i for i in range(10)]
    actual_data = [
        map_ds_no_transform[i] for i in range(len(map_ds_no_transform))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_map_data_with_transform(self):
    map_ds_with_transform = MapLazyMapDataset(self.range_ds, MapWithTransform())
    expected_data = [i + 1 for i in range(10)]
    actual_data = [
        map_ds_with_transform[i] for i in range(len(map_ds_with_transform))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_random_map_data_with_transform(self):
    map_ds_with_random_transform = MapLazyMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [
        map_ds_with_random_transform[i]
        for i in range(len(map_ds_with_random_transform))
    ]
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)


class MapLazyIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = lazy_dataset.RangeLazyMapDataset(
        0, 10
    ).to_iter_dataset()

  def test_map_data_no_transform(self):
    map_no_transform_iter_ds = iter(
        MapLazyIterDataset(self.range_iter_ds, MapWithNoTransform())
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(map_no_transform_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_map_data_with_transform(self):
    map_with_transform_iter_ds = iter(
        MapLazyIterDataset(self.range_iter_ds, MapWithTransform())
    )
    expected_data = [i + 1 for i in range(10)]
    actual_data = [next(map_with_transform_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_random_map_data_with_transform(self):
    map_with_random_transform_iter_ds = iter(
        MapLazyIterDataset(self.range_iter_ds, RandomMapWithTransform(), seed=0)
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(map_with_random_transform_iter_ds) for _ in range(10)]
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)

  def test_map_past_one_epoch_raises_exception(self):
    map_no_transform_iter_ds = iter(
        MapLazyIterDataset(self.range_iter_ds, MapWithNoTransform())
    )
    with self.assertRaises(StopIteration):
      next(map_no_transform_iter_ds)
      _ = [next(map_no_transform_iter_ds) for _ in range(20)]


if __name__ == "__main__":
  absltest.main()
