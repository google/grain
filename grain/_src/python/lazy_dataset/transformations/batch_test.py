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
"""Tests for batch transformation."""

from absl.testing import absltest
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import batch
import numpy as np


class BatchLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = lazy_dataset.RangeLazyMapDataset(0, 10)

  def test_batch_size_2(self):
    ds = batch.BatchLazyMapDataset(self.range_ds, batch_size=2)
    self.assertLen(ds, 5)  # 10 // 2 = 5.
    actual = [ds[i] for i in range(5)]
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    np.testing.assert_allclose(actual, expected)

  def test_batch_size_3(self):
    # drop_remainder defaults to False
    ds = batch.BatchLazyMapDataset(self.range_ds, batch_size=3)
    self.assertLen(ds, 4)  # ceil(10 / 3).
    actual = [ds[i] for i in range(4)]
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    for i in range(4):
      np.testing.assert_allclose(actual[i], expected[i])

  def test_batch_size_3_drop_remainder(self):
    ds = batch.BatchLazyMapDataset(
        self.range_ds, batch_size=3, drop_remainder=True
    )
    self.assertLen(ds, 3)  # 10 // 3.
    actual = [ds[i] for i in range(3)]
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    np.testing.assert_allclose(actual, expected)


class BatchLazyIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = lazy_dataset.RangeLazyMapDataset(0, 10).to_iter_dataset()

  def test_batch_size_2(self):
    ds = batch.BatchLazyIterDataset(self.range_ds, batch_size=2)
    ds_iter = iter(ds)
    actual = [next(ds_iter) for _ in range(5)]
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    np.testing.assert_allclose(actual, expected)

  def test_batch_size_3(self):
    # drop_remainder defaults to False
    ds = batch.BatchLazyIterDataset(self.range_ds, batch_size=3)
    actual = list(ds)
    self.assertLen(actual, 4)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    for i in range(4):
      np.testing.assert_allclose(actual[i], expected[i])

  def test_batch_size_3_drop_remainder(self):
    ds = batch.BatchLazyIterDataset(
        self.range_ds, batch_size=3, drop_remainder=True
    )
    actual = list(ds)
    self.assertLen(actual, 3)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
  absltest.main()
