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
"""Tests for filter transformation."""

import dataclasses
import itertools

from absl.testing import absltest
from grain._src.core.transforms import FilterTransform
from grain._src.python.lazy_dataset.lazy_dataset import RangeLazyMapDataset
from grain._src.python.lazy_dataset.transformations.filter import FilterLazyIterDataset
from grain._src.python.lazy_dataset.transformations.filter import FilterLazyMapDataset


@dataclasses.dataclass(frozen=True)
class FilterNoElements(FilterTransform):

  def filter(self, element: int):
    return True


@dataclasses.dataclass(frozen=True)
class FilterAllElements(FilterTransform):

  def filter(self, element: int):
    return False


@dataclasses.dataclass(frozen=True)
class FilterEvenElementsOnly(FilterTransform):

  def filter(self, element: int):
    return element % 2


class FilterLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = RangeLazyMapDataset(0, 10)

  def test_filter_size(self):
    filter_no_elts_ds = FilterLazyMapDataset(self.range_ds, FilterNoElements())
    filter_all_elts_ds = FilterLazyMapDataset(
        self.range_ds, FilterAllElements()
    )
    filter_even_elts_ds = FilterLazyMapDataset(
        self.range_ds, FilterEvenElementsOnly()
    )
    self.assertLen(filter_no_elts_ds, len(self.range_ds))
    self.assertLen(filter_all_elts_ds, len(self.range_ds))
    self.assertLen(filter_even_elts_ds, len(self.range_ds))

  def test_filter_no_elements(self):
    filter_no_elts_ds = FilterLazyMapDataset(self.range_ds, FilterNoElements())
    expected_data = [_ for _ in range(10)]
    actual_data = [filter_no_elts_ds[i] for i in range(len(filter_no_elts_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_filter_all_elements(self):
    filter_all_elts_ds = FilterLazyMapDataset(
        self.range_ds, FilterAllElements()
    )
    expected_data = [None for _ in range(10)]
    actual_data = [
        filter_all_elts_ds[i] for i in range(len(filter_all_elts_ds))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_filter_even_elements_only(self):
    filter_even_elts_ds = FilterLazyMapDataset(
        self.range_ds, FilterEvenElementsOnly()
    )
    expected_data = list(
        itertools.chain.from_iterable([[None, i + 1] for i in range(0, 10, 2)])
    )
    actual_data = [
        filter_even_elts_ds[i] for i in range(len(filter_even_elts_ds))
    ]
    self.assertEqual(expected_data, actual_data)


class FilterLazyIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = RangeLazyMapDataset(0, 10).to_iter_dataset()

  def test_filter_no_elements(self):
    filter_no_elts_iter_ds = iter(
        FilterLazyIterDataset(self.range_iter_ds, FilterNoElements())
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(filter_no_elts_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_filter_all_elements(self):
    filter_all_elts_iter_ds = iter(
        FilterLazyIterDataset(self.range_iter_ds, FilterAllElements())
    )
    with self.assertRaises(StopIteration):
      next(filter_all_elts_iter_ds)

  def test_filter_even_elements_only(self):
    filter_even_elts_iter_ds = iter(
        FilterLazyIterDataset(self.range_iter_ds, FilterEvenElementsOnly())
    )
    expected_data = [_ for _ in range(1, 10, 2)]
    actual_data = [next(filter_even_elts_iter_ds) for _ in range(5)]
    self.assertEqual(expected_data, actual_data)


if __name__ == "__main__":
  absltest.main()
