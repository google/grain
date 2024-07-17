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
from absl.testing import parameterized
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import shuffle


class ShuffleMapDatasetTest(parameterized.TestCase):

  def test_len(self):
    ds = shuffle.ShuffleMapDataset(lazy_dataset.RangeMapDataset(400), seed=42)
    self.assertLen(ds, 400)

  def test_getitem(self):
    ds = shuffle.ShuffleMapDataset(lazy_dataset.RangeMapDataset(400), seed=42)
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(set(shuffled_indices), 400)
    for x in shuffled_indices:
      self.assertBetween(x, 0, 400)
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_iter(self):
    ds = shuffle.ShuffleMapDataset(lazy_dataset.RangeMapDataset(400), seed=42)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    self.assertLen(elements, 400)

  @parameterized.parameters(-1000, -1, 2**32, 2**32 + 1, 2**64 + 1)
  def test_init_with_invalid_seed_returns_value_error(self, seed):
    with self.assertRaises(ValueError):
      shuffle.ShuffleMapDataset(lazy_dataset.RangeMapDataset(400), seed=seed)


class WindowShuffleMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = shuffle.WindowShuffleMapDataset(
        lazy_dataset.RangeMapDataset(400), window_size=10, seed=42
    )
    self.assertLen(ds, 400)

  def test_getitem(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        lazy_dataset.RangeMapDataset(400),
        window_size=window_size,
        seed=42,
    )
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(shuffled_indices, 400)
    for i in range(0, 400, 10):
      self.assertBetween(shuffled_indices[i], i, i + (window_size - 1))

  def test_getitem_multi_epochs(self):
    # Multiple epochs shouldn't affect window shuffling.
    ds = shuffle.WindowShuffleMapDataset(
        lazy_dataset.RangeMapDataset(400),
        window_size=10,
        seed=42,
    )
    shuffled_indices = [ds[i] for i in range(400)]
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertLen(shuffled_indices, 400)
    self.assertLen(shuffled_indices_epoch2, 400)
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_iter(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        lazy_dataset.RangeMapDataset(400), window_size=window_size, seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    for i in range(0, 400, 10):
      self.assertBetween(elements[i], i, i + (window_size - 1))


if __name__ == "__main__":
  absltest.main()
