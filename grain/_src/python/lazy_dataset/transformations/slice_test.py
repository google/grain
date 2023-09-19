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
"""Tests for slice transformation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.lazy_dataset import lazy_dataset
import grain._src.python.lazy_dataset.transformations.slice as slice_ds


class SliceLazyMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data_len = 20
    self.range_ds = lazy_dataset.RangeLazyMapDataset(self.data_len)
    self.range_py_list = list(range(self.data_len))

  @parameterized.parameters(
      (0, 1, 20),
      (0, 2, 10),
      (1, 2, 10),
      (0, 3, 7),
      (1, 3, 7),
      (2, 3, 6),
  )
  def test_len(self, start: int, step: int, expected_len: int):
    range_ds_for_process = slice_ds.SliceLazyMapDataset(
        self.range_ds, slice(start, self.data_len, step)
    )
    self.assertLen(range_ds_for_process, expected_len)

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_getitem(self, start: int, stop: int, step: int):
    expected = self.range_py_list[start:stop:step]
    ds = slice_ds.SliceLazyMapDataset(self.range_ds, slice(start, stop, step))
    actual = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_getitem_sice(self, start: int, stop: int, step: int):
    expected = self.range_py_list[start:stop:step]
    ds = self.range_ds[start:stop:step]
    actual = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_iter(self, start: int, stop: int, step: int):
    expected = self.range_py_list[start:stop:step]
    ds = slice_ds.SliceLazyMapDataset(self.range_ds, slice(start, stop, step))
    ds_iter = iter(ds)
    actual = list(ds_iter)
    self.assertSequenceEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
