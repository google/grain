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
"""Tests for flatmap transformation."""

import dataclasses
import itertools

from absl.testing import absltest
from grain._src.core import transforms
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import flatmap


@dataclasses.dataclass(frozen=True)
class FixedSizeSplitWithNoTransform(transforms.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element: int):
    for _ in range(self.max_fan_out):
      yield element


@dataclasses.dataclass(frozen=True)
class FixedSizeSplitWithTransform(transforms.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element: int):
    for _ in range(self.max_fan_out):
      yield element + 1


@dataclasses.dataclass(frozen=True)
class VariableSizeCappedSplitWithNoTransform(transforms.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element: int):
    for _ in range(min(element, self.max_fan_out)):
      yield element


@dataclasses.dataclass(frozen=True)
class VariableSizeUncappedSplitWithNoTransform(transforms.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element: int):
    for _ in range(element):
      yield element


class FlatMapLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = lazy_dataset.RangeLazyMapDataset(0, 10)
    self.fan_out = 10

  def test_fixed_fan_out_size(self):
    flatmap_ds = flatmap.FlatMapLazyMapDataset(
        self.range_ds, FixedSizeSplitWithNoTransform(max_fan_out=self.fan_out)
    )
    self.assertLen(flatmap_ds, self.fan_out * len(self.range_ds))

  def test_fixed_fan_out_data_no_transform(self):
    flatmap_ds = flatmap.FlatMapLazyMapDataset(
        self.range_ds, FixedSizeSplitWithNoTransform(max_fan_out=self.fan_out)
    )
    expected_data = list(
        itertools.chain.from_iterable([[i] * self.fan_out for i in range(10)])
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_fixed_fan_out_data_with_transform(self):
    flatmap_ds = flatmap.FlatMapLazyMapDataset(
        self.range_ds, FixedSizeSplitWithTransform(max_fan_out=self.fan_out)
    )
    expected_data = list(
        itertools.chain.from_iterable(
            [[i + 1] * self.fan_out for i in range(10)]
        )
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_variable_fan_out_size_still_fixed(self):
    flatmap_ds = flatmap.FlatMapLazyMapDataset(
        self.range_ds,
        VariableSizeCappedSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    self.assertLen(flatmap_ds, self.fan_out * len(self.range_ds))

  def test_variable_size_fan_out_data_has_nones(self):
    flatmap_ds = flatmap.FlatMapLazyMapDataset(
        self.range_ds,
        VariableSizeCappedSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    expected_data = list(
        itertools.chain.from_iterable(
            [[i] * i + [None] * (self.fan_out - i) for i in range(10)]
        )
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_fan_out_exceeds_max_size_raises_error(self):
    with self.assertRaises(ValueError):
      longer_range_ds = lazy_dataset.RangeLazyMapDataset(0, 20)
      flatmap_ds = flatmap.FlatMapLazyMapDataset(
          longer_range_ds,
          VariableSizeUncappedSplitWithNoTransform(max_fan_out=self.fan_out),
      )
      _ = [flatmap_ds[i] for i in range(len(flatmap_ds))]


if __name__ == "__main__":
  absltest.main()
