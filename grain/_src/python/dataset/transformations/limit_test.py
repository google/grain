# Copyright 2025 Google LLC
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
"""Tests for limit transformations."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import limit
import grain._src.python.testing.experimental as test_util


class LimitIterDatasetTest(parameterized.TestCase):

  @parameterized.parameters([0, -1, -5])
  def test_non_positive_count_raises_error(self, count):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    with self.assertRaises(ValueError):
      _ = limit.LimitIterDataset(ds, count=count)

  def test_stop_iteration_raised_after_limit_reached(self):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    ds = limit.LimitIterDataset(ds, count=1)
    ds_iter = iter(ds)
    _ = next(ds_iter)
    with self.assertRaises(StopIteration):
      next(ds_iter)

  @parameterized.parameters([1, 3, 5, 7, 10])
  def test_count(self, count):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    ds = limit.LimitIterDataset(ds, count=count)
    actual_data = list(ds)
    self.assertLen(actual_data, count)
    self.assertEqual(actual_data, list(range(count)))

  def test_count_over_epochs(self):
    ds = dataset.MapDataset.range(0, 10).repeat(2).to_iter_dataset()
    ds = limit.LimitIterDataset(ds, count=15)
    actual_data = list(ds)
    self.assertLen(actual_data, 15)
    self.assertEqual(actual_data, list(range(10)) + list(range(5)))

  def test_limit_after_batch(self):
    def flatten_batches(batches):
      actual_data = []
      for batch in batches:
        actual_data.extend(batch.tolist())
      return actual_data

    ds = dataset.MapDataset.range(0, 10).batch(3).to_iter_dataset()

    ds_1 = limit.LimitIterDataset(ds, count=2)
    batches = list(ds_1)
    actual_data = flatten_batches(batches)
    self.assertEqual(actual_data, list(range(6)))

    ds_2 = limit.LimitIterDataset(ds, count=5)
    batches = list(ds_2)
    actual_data = flatten_batches(batches)
    self.assertLen(batches, 4)
    self.assertEqual(actual_data, list(range(10)))

  def test_checkpointing(self):
    ds = dataset.MapDataset.range(0, 10).batch(3).to_iter_dataset()
    limited_ds = limit.LimitIterDataset(ds, count=2)
    test_util.assert_equal_output_after_checkpoint(limited_ds)


if __name__ == "__main__":
  absltest.main()
