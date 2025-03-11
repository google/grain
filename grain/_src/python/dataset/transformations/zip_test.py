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
from grain._src.python.dataset import dataset
import grain._src.python.dataset.transformations.zip as zip_ds
import grain._src.python.testing.experimental as test_util


class ZipMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_list = [
        dataset.MapDataset.range(0, 20),
        dataset.MapDataset.range(1, 21),
        dataset.MapDataset.range(2, 22),
    ]

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in list(itertools.combinations(range(3), 3))
      + list(itertools.combinations(range(3), 2))
      + list(itertools.combinations(range(3), 1))
  )
  def test_len(self, ds_idx_list):
    self.assertLen(
        zip_ds.ZipMapDataset(parents=[self.ds_list[i] for i in ds_idx_list]),
        20,
    )

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in list(itertools.combinations(range(3), 3))
      + list(itertools.combinations(range(3), 2))
      + list(itertools.combinations(range(3), 1))
  )
  def test_getitem(self, ds_idx_list):
    ds = zip_ds.ZipMapDataset(parents=[self.ds_list[i] for i in ds_idx_list])
    for i in range(20):
      self.assertEqual(ds[i], tuple(i + ds_idx for ds_idx in ds_idx_list))


class ZipIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_list = [
        dataset.MapDataset.range(0, 20),
        dataset.MapDataset.range(1, 21),
        dataset.MapDataset.range(2, 22),
    ]

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in list(itertools.combinations(range(3), 3))
      + list(itertools.combinations(range(3), 2))
      + list(itertools.combinations(range(3), 1))
  )
  def test_iter(self, ds_idx_list):
    ds = zip_ds.ZipIterDataset(
        parents=[self.ds_list[i].to_iter_dataset() for i in ds_idx_list]
    )
    out = list(ds)
    for i in range(20):
      self.assertEqual(out[i], tuple(i + ds_idx for ds_idx in ds_idx_list))

  def test_strict_zip_shorter(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 2 is shorter than argument 1"
    ):
      list(ds)

  def test_strict_zip_shorter_many(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 3 is shorter than arguments 1-2"
    ):
      list(ds)

  def test_strict_zip_longer_many(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(3).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 4 is longer than arguments 1-3"
    ):
      list(ds)

  def test_non_strict_zip(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(1, 4).to_iter_dataset(),
        ],
        strict=False,
    )
    actual = list(ds)
    expected = [(0, 1), (1, 2)]
    self.assertEqual(actual, expected)

  def test_checkpointing(self):
    ds = zip_ds.ZipIterDataset(
        parents=[p.to_iter_dataset() for p in self.ds_list]
    )
    test_util.assert_equal_output_after_checkpoint(ds)


if __name__ == "__main__":
  absltest.main()
