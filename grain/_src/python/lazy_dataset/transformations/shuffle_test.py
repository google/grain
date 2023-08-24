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
"""Tests for shuffle transformation."""

from absl.testing import absltest
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import shuffle


class ShuffleLazyMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = shuffle.ShuffleLazyMapDataset(
        lazy_dataset.RangeLazyMapDataset(400), seed=42
    )
    self.assertLen(ds, 400)

  def test_getitem(self):
    ds = shuffle.ShuffleLazyMapDataset(
        lazy_dataset.RangeLazyMapDataset(400),
        reshuffle_each_epoch=False,
        seed=42,
    )
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(set(shuffled_indices), 400)
    for x in shuffled_indices:
      self.assertBetween(x, 0, 400)
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_getitem_reshuffle(self):
    ds = shuffle.ShuffleLazyMapDataset(
        lazy_dataset.RangeLazyMapDataset(400), seed=42
    )
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(set(shuffled_indices), 400)
    for x in shuffled_indices:
      self.assertBetween(x, 0, 400)
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_iter(self):
    ds = shuffle.ShuffleLazyMapDataset(
        lazy_dataset.RangeLazyMapDataset(400), seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    self.assertLen(elements, 400)


if __name__ == "__main__":
  absltest.main()
