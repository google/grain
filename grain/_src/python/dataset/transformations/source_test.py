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
"""Tests for LazyDataset data sources."""
import random
from unittest import mock

from absl.testing import absltest
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import source


class _Interleave(dataset.MapDataset):

  def __len__(self):
    return sum((len(p) for p in self.parents))

  def __getitem__(self, index):
    index, parent_index = divmod(index, len(self.parents))
    return self.parents[parent_index][index]


class SourceMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_data_source = [1, 2, 3, 4, 5]
    self.lazy_dataset_source = source.SourceMapDataset(  # pytype: disable=wrong-arg-types
        self.sample_data_source
    )

  def test_lazy_dataset_source_len(self):
    self.assertLen(self.lazy_dataset_source, 5)

  def test_lazy_dataset_source_sequential_get(self):
    indices_to_read = [0, 1, 2, 3, 4]
    expected_data = [1, 2, 3, 4, 5]
    actual_data = [self.lazy_dataset_source[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_lazy_dataset_source_reverse_sequential_get(self):
    indices_to_read = [0, 1, 2, 3, 4]
    expected_data = [1, 2, 3, 4, 5]
    indices_to_read.reverse()
    expected_data.reverse()
    actual_data = [self.lazy_dataset_source[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_lazy_dataset_source_random_get(self):
    indices_to_read = [0, 1, 2, 3, 4]
    random.shuffle(indices_to_read)
    expected_data = [self.sample_data_source[i] for i in indices_to_read]
    actual_data = [self.lazy_dataset_source[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_lazy_dataset_source_random_modulo_get(self):
    len_data_source = len(self.lazy_dataset_source)
    indices_to_read = [100, 207, 303, 401]
    expected_data = [
        self.sample_data_source[i % len_data_source] for i in indices_to_read
    ]
    actual_data = [self.lazy_dataset_source[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)


class RangeMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = source.RangeMapDataset(12)
    self.assertLen(ds, 12)
    ds = source.RangeMapDataset(0, 12)
    self.assertLen(ds, 12)
    ds = source.RangeMapDataset(2, 12)
    self.assertLen(ds, 10)
    ds = source.RangeMapDataset(2, 12, 1)
    self.assertLen(ds, 10)
    ds = source.RangeMapDataset(2, 12, 2)
    self.assertLen(ds, 5)
    ds = source.RangeMapDataset(2, 13, 2)
    self.assertLen(ds, 6)

  def test_getitem(self):
    ds = source.RangeMapDataset(12)
    for i in range(12):
      self.assertEqual(ds[i], i)
    for i in range(12):
      self.assertEqual(ds[i + 12], i)
    ds = source.RangeMapDataset(2, 9, 2)
    self.assertEqual(ds[0], 2)
    self.assertEqual(ds[1], 4)
    self.assertEqual(ds[2], 6)
    self.assertEqual(ds[3], 8)
    self.assertEqual(ds[4], 2)
    self.assertEqual(ds[5], 4)

  def test_iter(self):
    ds = source.RangeMapDataset(12)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(12)]
    self.assertEqual(elements, list(range(12)))
    ds = source.RangeMapDataset(2, 9, 2)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(4)]
    self.assertEqual(elements, [2, 4, 6, 8])


if __name__ == "__main__":
  absltest.main()
