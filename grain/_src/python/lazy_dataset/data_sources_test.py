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
from grain._src.python.lazy_dataset import data_sources
from grain._src.python.lazy_dataset import lazy_dataset


class _Interleave(lazy_dataset.LazyMapDataset):

  def __len__(self):
    return sum((len(p) for p in self.parents))

  def __getitem__(self, index):
    index, parent_index = divmod(index, len(self.parents))
    return self.parents[parent_index][index]


class SourceLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_data_source = [1, 2, 3, 4, 5]
    self.lazy_dataset_source = data_sources.SourceLazyMapDataset(  # pytype: disable=wrong-arg-types
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


if __name__ == "__main__":
  absltest.main()
