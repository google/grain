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

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_matches_python_slicing(self, start: int, stop: int, step: int):
    expected = list(range(6))[start:stop:step]
    ds = lazy_dataset.RangeLazyMapDataset(6)
    ds = slice_ds.SliceLazyMapDataset(ds, start=start, stop=stop, step=step)
    self.assertLen(ds, len(expected))
    actual = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(actual, expected)


if __name__ == "__main__":
  absltest.main()
