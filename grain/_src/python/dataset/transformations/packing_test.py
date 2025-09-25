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
"""Tests for batch transformation."""
from typing import Any

from absl.testing import absltest
from grain._src.python.dataset.transformations import packing
from grain._src.python.dataset.transformations import source
from grain._src.python.dataset.transformations import testing_util
import numpy as np


# A "mixin" class to provide the common memory size tests.
# It is not a TestCase itself but provides test methods to other classes.
# It assumes that any class using it will define `self.packer_cls`.
class _PackedBatchSizeBytesTestMixin:
  """A mixin providing common tests for get_packed_batch_size_bytes."""

  kwargs: dict[str, Any]

  def test_get_packed_batch_size_bytes(self):
    ds = source.SourceMapDataset([
        {"x": np.zeros(5, dtype=np.int64)},
        {"x": np.ones(4, dtype=np.int64)},
        {"x": np.zeros(10, dtype=np.int64)},
    ]).to_iter_dataset()
    ds = packing.FirstFitPackIterDataset(
        ds,
        length_struct={"x": 10},
        num_packing_bins=2,
        **self.kwargs,
    )
    iterator = ds.__iter__()
    # Get one element to initialize packing.
    next(iterator)
    # 2*10*8 (values) + 2*10*4 (segment_ids) + 2*10*4 (positions) + 2*8
    # (first_free_cell) = 160 + 80 + 80 + 16 = 336
    self.assertEqual(iterator.get_packed_batch_size_bytes(), 336)  # pytype: disable=attribute-error

  def test_get_packed_batch_size_bytes_before_next(self):
    ds = source.SourceMapDataset([
        {"x": np.zeros(5, dtype=np.int64)},
    ]).to_iter_dataset()
    ds = packing.FirstFitPackIterDataset(
        ds,
        length_struct={"x": 10},
        num_packing_bins=2,
        **self.kwargs,
    )
    iterator = ds.__iter__()
    # Check size before calling next()
    self.assertRaises(ValueError, iterator.get_packed_batch_size_bytes)  # pytype: disable=attribute-error


class FirstFitPackIterDatasetTest(
    _PackedBatchSizeBytesTestMixin, testing_util.BaseFirstFitPackIterDatasetTest
):

  def setUp(self):
    super().setUp()
    self.kwargs = {}


class BestFitPackIterDatasetTest(
    _PackedBatchSizeBytesTestMixin, testing_util.BaseBestFitPackIterDatasetTest
):

  def setUp(self):
    super().setUp()
    self.kwargs = {}

if __name__ == "__main__":
  absltest.main()
