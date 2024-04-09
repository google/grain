# Copyright 2024 Google LLC
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
"""Tests for zip transformation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.lazy_dataset import lazy_dataset
import grain._src.python.lazy_dataset.transformations.zip as zip_ds


class ZipLazyMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_list = [
        lazy_dataset.RangeLazyMapDataset(0, 20),
        lazy_dataset.RangeLazyMapDataset(1, 21),
        lazy_dataset.RangeLazyMapDataset(2, 22),
    ]

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in list(itertools.combinations(range(3), 3))
      + list(itertools.combinations(range(3), 2))
      + list(itertools.combinations(range(3), 1))
  )
  def test_len(self, ds_idx_list):
    self.assertLen(
        zip_ds.ZipLazyMapDataset(
            parents=(self.ds_list[i] for i in ds_idx_list)
        ),
        20,
    )

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in list(itertools.combinations(range(3), 3))
      + list(itertools.combinations(range(3), 2))
      + list(itertools.combinations(range(3), 1))
  )
  def test_getitem(self, ds_idx_list):
    ds = zip_ds.ZipLazyMapDataset(
        parents=(self.ds_list[i] for i in ds_idx_list)
    )
    for i in range(20):
      self.assertEqual(ds[i], tuple(i + ds_idx for ds_idx in ds_idx_list))


if __name__ == "__main__":
  absltest.main()
