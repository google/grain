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


class RepeatLazyMapDatasetTest(absltest.TestCase):

  def test_fixed_num_epochs(self):
    ds = lazy_dataset.RangeLazyMapDataset(6)
    self.assertLen(ds, 6)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=3)
    self.assertLen(ds, 3 * 6)
    self.assertEqual(list(ds), 3 * list(range(6)))

  def test_infinite_epochs(self):
    ds = lazy_dataset.RangeLazyMapDataset(6)
    ds = repeat.RepeatLazyMapDataset(ds, num_epochs=None)
    self.assertLen(ds, sys.maxsize)
    # Repeating again fails.
    with self.assertRaises(ValueError):
      repeat.RepeatLazyMapDataset(ds, num_epochs=2)


if __name__ == "__main__":
  absltest.main()
