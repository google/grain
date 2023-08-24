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
"""Tests for ragged batch transformation."""

import dataclasses
from typing import Sequence

from absl.testing import absltest
from grain._src.core.transforms import MapTransform
from grain._src.core.transforms import RaggedBatchTransform
from grain._src.python.lazy_dataset.data_sources import SourceLazyMapDataset
from grain._src.python.lazy_dataset.lazy_dataset import RangeLazyMapDataset
from grain._src.python.lazy_dataset.transformations.map import MapLazyMapDataset
from grain._src.python.lazy_dataset.transformations.ragged_batch import RaggedBatchLazyMapDataset


@dataclasses.dataclass(frozen=True)
class IntegerRaggedBatchMapTransform(MapTransform):

  def map(self, elements: Sequence[int]) -> Sequence[int]:
    return [elt + 1 for elt in elements]


@dataclasses.dataclass(frozen=True)
class TextRaggedBatchMapTransform(MapTransform):

  def map(self, elements: Sequence[str]) -> str:
    return " ".join(elements)


class RaggedBatchTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = RangeLazyMapDataset(0, 10)

  def test_concatenate_single_element_no_transform(self):
    ragged_batch_ds = RaggedBatchLazyMapDataset(
        self.range_ds, RaggedBatchTransform(batch_size=1)
    )
    expected_data = [[i] for i in range(10)]
    actual_data = [ragged_batch_ds[i] for i in range(len(ragged_batch_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_concatenate_multi_element_no_transform(self):
    ragged_batch_ds = RaggedBatchLazyMapDataset(
        self.range_ds, RaggedBatchTransform(batch_size=2)
    )
    expected_data = [[i, i + 1] for i in range(0, 10, 2)]
    actual_data = [ragged_batch_ds[i] for i in range(len(ragged_batch_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_concatenate_multi_element_no_transform_hanging_end(self):
    longer_range_ds = RangeLazyMapDataset(0, 11)
    ragged_batch_ds = RaggedBatchLazyMapDataset(
        longer_range_ds, RaggedBatchTransform(batch_size=2)
    )
    expected_data = [[i, i + 1] for i in range(0, 10, 2)] + [[10]]
    actual_data = [ragged_batch_ds[i] for i in range(len(ragged_batch_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_concatenate_multi_element_with_transform(self):
    ragged_batch_ds = RaggedBatchLazyMapDataset(
        self.range_ds, RaggedBatchTransform(batch_size=2)
    )
    ragged_batch_ds_with_transform = MapLazyMapDataset(
        ragged_batch_ds, IntegerRaggedBatchMapTransform()
    )
    expected_data = [[i + 1, i + 2] for i in range(0, 10, 2)]
    actual_data = [
        ragged_batch_ds_with_transform[i]
        for i in range(len(ragged_batch_ds_with_transform))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_concatenate_multi_element_with_text_transform(self):
    range_text_ds = SourceLazyMapDataset(
        [str(self.range_ds[i]) for i in range(len(self.range_ds))]
    )
    ragged_batch_ds = RaggedBatchLazyMapDataset(
        range_text_ds, RaggedBatchTransform(batch_size=2)
    )
    ragged_batch_ds_with_transform = MapLazyMapDataset(
        ragged_batch_ds, TextRaggedBatchMapTransform()
    )
    expected_data = ["{} {}".format(i, i + 1) for i in range(0, 10, 2)]
    actual_data = [
        ragged_batch_ds_with_transform[i]
        for i in range(len(ragged_batch_ds_with_transform))
    ]
    self.assertEqual(expected_data, actual_data)


if __name__ == "__main__":
  absltest.main()
