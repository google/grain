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
"""Tests for rebatch transformation."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import batch
from grain._src.python.dataset.transformations import rebatch
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np
import tree


class RebatchIterDatasetTest(parameterized.TestCase):

  def _assert_trees_equal(self, actual, expected):
    def _check_equivalence(path, actual_val, expected_val):
      np.testing.assert_array_equal(
          actual_val,
          expected_val,
          err_msg=(
              f"Pytrees differ at path {path}.\n\n"
              f"Actual: {actual_val}\n\nExpected: {expected_val}"
          ),
      )

    tree.map_structure_with_path(_check_equivalence, actual, expected)

  def _get_test_tree_element(self, x: int) -> dict[str, np.ndarray]:
    element = {}
    # Simulates having 5 features in a single dictionary.
    for i in range(5):
      feature = np.full(3, x)
      key = f"key_{i}"
      element[key] = feature
    return element

  def _get_expected_tree(self, values: list[int]) -> dict[str, np.ndarray]:
    pytree = {}
    for i in range(5):
      key = f"key_{i}"
      pytree[key] = np.array([np.full(3, x) for x in values])
    return pytree

  def _get_test_dataset_ten_elements(
      self, batch_size, rebatch_size, drop_remainder=False
  ):
    ds = dataset.MapDataset.range(0, 10)
    ds = ds.map(self._get_test_tree_element)
    ds = ds.to_iter_dataset()
    ds = batch.BatchIterDataset(ds, batch_size=batch_size)
    ds = rebatch.RebatchIterDataset(
        ds, batch_size=rebatch_size, drop_remainder=drop_remainder
    )
    return ds

  @parameterized.named_parameters(
      dict(
          testcase_name="to_larger_batch",
          batch_size=2,
          rebatch_size=4,
          drop_remainder=False,
          expected_elements_by_batch=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],
      ),
      dict(
          testcase_name="to_smaller_batch",
          batch_size=4,
          rebatch_size=2,
          drop_remainder=False,
          expected_elements_by_batch=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
      ),
      dict(
          testcase_name="to_same_batch",
          batch_size=3,
          rebatch_size=3,
          drop_remainder=False,
          expected_elements_by_batch=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
      ),
      dict(
          testcase_name="uneven_batches",
          batch_size=3,
          rebatch_size=4,
          drop_remainder=False,
          expected_elements_by_batch=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]],
      ),
      dict(
          testcase_name="to_larger_batch_drop_remainder",
          batch_size=2,
          rebatch_size=4,
          drop_remainder=True,
          expected_elements_by_batch=[[0, 1, 2, 3], [4, 5, 6, 7]],
      ),
  )
  def test_rebatch(
      self, batch_size, rebatch_size, drop_remainder, expected_elements_by_batch
  ):
    ds = self._get_test_dataset_ten_elements(
        batch_size=batch_size,
        rebatch_size=rebatch_size,
        drop_remainder=drop_remainder,
    )
    actual_elements = list(ds)
    expected_elements = [
        self._get_expected_tree(vals) for vals in expected_elements_by_batch
    ]
    self.assertLen(actual_elements, len(expected_elements))
    for actual, expected in zip(actual_elements, expected_elements):
      self._assert_trees_equal(actual, expected)

  def test_rebatch_checkpointing(self):
    ds = self._get_test_dataset_ten_elements(batch_size=3, rebatch_size=4)
    assert_equal_output_after_checkpoint(ds)
    ds = self._get_test_dataset_ten_elements(batch_size=4, rebatch_size=3)
    assert_equal_output_after_checkpoint(ds)
    ds = self._get_test_dataset_ten_elements(batch_size=4, rebatch_size=4)
    assert_equal_output_after_checkpoint(ds)

  def test_stop_iteration_exhausted_not_divisible_batch_size(self):
    ds = self._get_test_dataset_ten_elements(batch_size=3, rebatch_size=4)
    ds_iter = iter(ds)
    for _ in range(3):
      next(ds_iter)
    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_stop_iteration_exhausted_divisible_batch_size(self):
    ds = self._get_test_dataset_ten_elements(batch_size=4, rebatch_size=2)
    ds_iter = iter(ds)
    for _ in range(5):
      next(ds_iter)
    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_negative_batch_size(self):
    with self.assertRaises(ValueError):
      self._get_test_dataset_ten_elements(batch_size=2, rebatch_size=-1)


if __name__ == "__main__":
  absltest.main()
