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
from grain._src.core import transforms
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import filter as filter_dataset


@dataclasses.dataclass(frozen=True)
class FilterNoElements(transforms.Filter):

  def filter(self, element: int):
    return True


@dataclasses.dataclass(frozen=True)
class FilterAllElements(transforms.Filter):

  def filter(self, element: int):
    return False


@dataclasses.dataclass(frozen=True)
class FilterEvenElementsOnly(transforms.Filter):

  def filter(self, element: int):
    return element % 2


class FilterMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)

  def test_filter_size(self):
    filter_no_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterNoElements()
    )
    filter_all_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterAllElements()
    )
    filter_even_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterEvenElementsOnly()
    )
    self.assertLen(filter_no_elts_ds, len(self.range_ds))
    self.assertLen(filter_all_elts_ds, len(self.range_ds))
    self.assertLen(filter_even_elts_ds, len(self.range_ds))

  def test_filter_no_elements(self):
    filter_no_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterNoElements()
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [filter_no_elts_ds[i] for i in range(len(filter_no_elts_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_filter_all_elements(self):
    filter_all_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterAllElements()
    )
    expected_data = [None for _ in range(10)]
    actual_data = [
        filter_all_elts_ds[i] for i in range(len(filter_all_elts_ds))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_filter_even_elements_only(self):
    filter_even_elts_ds = filter_dataset.FilterMapDataset(
        self.range_ds, FilterEvenElementsOnly()
    )
    expected_data = list(
        itertools.chain.from_iterable([[None, i + 1] for i in range(0, 10, 2)])
    )
    actual_data = [
        filter_even_elts_ds[i] for i in range(len(filter_even_elts_ds))
    ]
    self.assertEqual(expected_data, actual_data)


class FilterIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    # Issue warnings without wait.
    filter_dataset._WARN_FILTERED_INTERVAL_SEC = 0.0

  def test_filter_no_elements(self):
    filter_no_elts_iter_ds = iter(
        filter_dataset.FilterIterDataset(self.range_iter_ds, FilterNoElements())
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(filter_no_elts_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_filter_all_elements(self):
    filter_all_elts_iter_ds = iter(
        filter_dataset.FilterIterDataset(
            self.range_iter_ds, FilterAllElements()
        )
    )
    with self.assertRaises(StopIteration):
      next(filter_all_elts_iter_ds)

  def test_filter_even_elements_only(self):
    filter_even_elts_iter_ds = iter(
        filter_dataset.FilterIterDataset(
            self.range_iter_ds, FilterEvenElementsOnly()
        )
    )
    expected_data = [_ for _ in range(1, 10, 2)]
    actual_data = [next(filter_even_elts_iter_ds) for _ in range(5)]
    self.assertEqual(expected_data, actual_data)

  def test_filter_all_elements_warns(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .to_iter_dataset()
        .filter(FilterAllElements())
    )
    with self.assertLogs(level="WARNING") as logs:
      _ = list(ds)
    logs = logs[0][0].message
    self.assertRegex(
        logs,
        r"Transformation FilterDatasetIterator\(transform=FilterAllElements\)"
        r" skipped 100.00 \% of the last seen 1000 elements.",
    )

  def test_filter_all_elements_raises(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .to_iter_dataset()
        .filter(FilterAllElements())
    )
    ds = dataset.WithOptionsIterDataset(
        ds, base.DatasetOptions(filter_raise_threshold_ratio=0.999)
    )
    with self.assertRaisesRegex(
        ValueError,
        r"Transformation FilterDatasetIterator\(transform=FilterAllElements\)"
        r" skipped 100.00 \% of the last seen 1000 elements.",
    ):
      _ = list(ds)


class FilterValidatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    filter_dataset._WARN_FILTERED_INTERVAL_SEC = 0.0

  def test_validates(self):
    default_options = base.DatasetOptions()
    v = filter_dataset.FilterThresholdChecker(
        "test",
        default_options.filter_warn_threshold_ratio,
        default_options.filter_raise_threshold_ratio,
    )
    passed = [True] * 101 + [False] * 899 + [True] * 100 + [False] * 900
    for p in passed:
      v.check(p)

  def test_warns(self):
    default_options = base.DatasetOptions()
    v = filter_dataset.FilterThresholdChecker(
        "test",
        default_options.filter_warn_threshold_ratio,
        default_options.filter_raise_threshold_ratio,
    )
    passed = [True] + [False] * filter_dataset._CHECK_FILTERED_INTERVAL
    with self.assertLogs(level="WARNING") as logs:
      for p in passed:
        v.check(p)
    logs = logs[0][0].message
    self.assertRegex(
        logs,
        r"Transformation test skipped 99.90 \% of the last seen 1000"
        r" elements.",
    )

  def test_raises(self):
    v = filter_dataset.FilterThresholdChecker("test", None, 0.999)
    passed = [False] * filter_dataset._CHECK_FILTERED_INTERVAL
    with self.assertRaisesRegex(
        ValueError,
        r"Transformation test skipped 100.00 \% of the last seen 1000"
        r" elements.",
    ):
      for p in passed:
        v.check(p)


if __name__ == "__main__":
  absltest.main()
