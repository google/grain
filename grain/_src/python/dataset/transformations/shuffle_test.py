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
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import shuffle


class ShuffleMapDatasetTest(parameterized.TestCase):

  def test_len(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    self.assertLen(ds, 400)

  def test_getitem(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(set(shuffled_indices), 400)
    for x in shuffled_indices:
      self.assertBetween(x, 0, 400)
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_iter(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    self.assertLen(elements, 400)

  def test_default_seed(self):
    seed = 42
    ds1 = dataset.MapDataset.range(400).seed(seed)
    ds1 = shuffle.ShuffleMapDataset(ds1)
    ds2 = dataset.MapDataset.range(400).seed(seed)
    ds2 = shuffle.ShuffleMapDataset(ds2)
    self.assertEqual(list(ds1), list(ds2))

  def test_raises_with_no_seed(self):
    with self.assertRaises(ValueError):
      shuffle.ShuffleMapDataset(dataset.MapDataset.range(400))

  @parameterized.parameters(-1000, -1, 2**32, 2**32 + 1, 2**64 + 1)
  def test_init_with_invalid_seed_returns_value_error(self, seed):
    with self.assertRaises(ValueError):
      shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=seed)


class WindowShuffleMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400), window_size=10, seed=42
    )
    self.assertLen(ds, 400)

  def test_getitem(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400),
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
        dataset.MapDataset.range(400),
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
        dataset.MapDataset.range(400), window_size=window_size, seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    for i in range(0, 400, 10):
      self.assertBetween(elements[i], i, i + (window_size - 1))


class WindowShuffleInterDatasetTest(absltest.TestCase):
  _DATASET_SIZE = 30
  _WINDOW_SIZE = 10

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(
        0, self._DATASET_SIZE
    ).to_iter_dataset()

  def test_shuffle_range(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=self._WINDOW_SIZE, seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(30)]
    for i in range(0, self._DATASET_SIZE, self._WINDOW_SIZE):
      self.assertBetween(elements[i], i, i + (self._WINDOW_SIZE - 1))

  def test_shuffle_range_within_window(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=self._WINDOW_SIZE, seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(30)]
    num_windows = int(self._DATASET_SIZE / self._WINDOW_SIZE)
    for window_num in range(0, num_windows):
      for pos_in_window in range(0, self._WINDOW_SIZE):
        first_pos_in_window = window_num * self._WINDOW_SIZE
        last_pos_in_window = (
            window_num * self._WINDOW_SIZE + self._WINDOW_SIZE - 1
        )
        self.assertBetween(
            elements[window_num * self._WINDOW_SIZE + pos_in_window],
            first_pos_in_window,
            last_pos_in_window,
        )


if __name__ == "__main__":
  absltest.main()
