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
"""Tests for repeat transformation."""
import sys

from absl.testing import absltest
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import repeat
from typing_extensions import override


class EmptyLazyMapDataset(lazy_dataset.LazyMapDataset[int]):

  def __init__(self):
    super().__init__(parents=[])

  @override
  def __len__(self) -> int:
    return 0

  @override
  def __getitem__(self, index):
    raise IndexError("Index out of range")


class RepeatLazyMapDatasetTest(absltest.TestCase):

  def test_finite_num_epochs_changes_length(self):
    ds = lazy_dataset.RangeLazyMapDataset(6)
    self.assertLen(ds, 6)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=3)
    self.assertLen(ds, 18)

  def test_finite_num_epochs_produces_expected_elements_when_iterated(self):
    ds = lazy_dataset.RangeLazyMapDataset(4)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=3)
    self.assertSequenceEqual(list(ds), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

  def test_infinite_epochs_sets_length_to_maxsize(self):
    ds = lazy_dataset.RangeLazyMapDataset(6)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=None)
    self.assertLen(ds, sys.maxsize)

  def test_repeat_after_setting_infinite_epochs_raises_value_error(self):
    ds = lazy_dataset.RangeLazyMapDataset(6)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=None)
    with self.assertRaises(ValueError):
      repeat.RepeatLazyMapDataset(ds, num_epochs=2)

  def test_infinite_epochs_of_empty_dataset_keeps_length_zero(self):
    ds = EmptyLazyMapDataset()
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=None)
    self.assertEmpty(ds)


if __name__ == "__main__":
  absltest.main()
