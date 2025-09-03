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

  def _get_test_dataset_ten_elements(
      self, batch_size, rebatch_size, drop_remainder=False
  ):
    def _get_test_tree_element(x: int) -> dict[str, np.ndarray]:
      element = {}
      # Simulates having 3 features in a single dictionary.
      for i in range(3):
        feature = np.full(2, x)
        key = f"key_{i}"
        element[key] = feature
      return element

    ds = dataset.MapDataset.range(0, 10)
    ds = ds.map(_get_test_tree_element)
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
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_1=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_2=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
              ),
              dict(
                  key_0=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_1=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_2=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
              ),
              dict(
                  key_0=np.array([[8, 8], [9, 9]]),
                  key_1=np.array([[8, 8], [9, 9]]),
                  key_2=np.array([[8, 8], [9, 9]]),
              ),
          ],
      ),
      dict(
          testcase_name="to_smaller_batch",
          batch_size=4,
          rebatch_size=2,
          drop_remainder=False,
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1]]),
                  key_1=np.array([[0, 0], [1, 1]]),
                  key_2=np.array([[0, 0], [1, 1]]),
              ),
              dict(
                  key_0=np.array([[2, 2], [3, 3]]),
                  key_1=np.array([[2, 2], [3, 3]]),
                  key_2=np.array([[2, 2], [3, 3]]),
              ),
              dict(
                  key_0=np.array([[4, 4], [5, 5]]),
                  key_1=np.array([[4, 4], [5, 5]]),
                  key_2=np.array([[4, 4], [5, 5]]),
              ),
              dict(
                  key_0=np.array([[6, 6], [7, 7]]),
                  key_1=np.array([[6, 6], [7, 7]]),
                  key_2=np.array([[6, 6], [7, 7]]),
              ),
              dict(
                  key_0=np.array([[8, 8], [9, 9]]),
                  key_1=np.array([[8, 8], [9, 9]]),
                  key_2=np.array([[8, 8], [9, 9]]),
              ),
          ],
      ),
      dict(
          testcase_name="to_same_batch",
          batch_size=3,
          rebatch_size=3,
          drop_remainder=False,
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1], [2, 2]]),
                  key_1=np.array([[0, 0], [1, 1], [2, 2]]),
                  key_2=np.array([[0, 0], [1, 1], [2, 2]]),
              ),
              dict(
                  key_0=np.array([[3, 3], [4, 4], [5, 5]]),
                  key_1=np.array([[3, 3], [4, 4], [5, 5]]),
                  key_2=np.array([[3, 3], [4, 4], [5, 5]]),
              ),
              dict(
                  key_0=np.array([[6, 6], [7, 7], [8, 8]]),
                  key_1=np.array([[6, 6], [7, 7], [8, 8]]),
                  key_2=np.array([[6, 6], [7, 7], [8, 8]]),
              ),
              dict(
                  key_0=np.array([[9, 9]]),
                  key_1=np.array([[9, 9]]),
                  key_2=np.array([[9, 9]]),
              ),
          ],
      ),
      dict(
          testcase_name="uneven_batches",
          batch_size=3,
          rebatch_size=4,
          drop_remainder=False,
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_1=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_2=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
              ),
              dict(
                  key_0=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_1=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_2=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
              ),
              dict(
                  key_0=np.array([[8, 8], [9, 9]]),
                  key_1=np.array([[8, 8], [9, 9]]),
                  key_2=np.array([[8, 8], [9, 9]]),
              ),
          ],
      ),
      dict(
          testcase_name="to_larger_batch_drop_remainder",
          batch_size=2,
          rebatch_size=4,
          drop_remainder=True,
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_1=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                  key_2=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
              ),
              dict(
                  key_0=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_1=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
                  key_2=np.array([[4, 4], [5, 5], [6, 6], [7, 7]]),
              ),
          ],
      ),
      dict(
          testcase_name="to_smaller_batch_drop_remainder",
          batch_size=4,
          rebatch_size=3,
          drop_remainder=True,
          expected_elements=[
              dict(
                  key_0=np.array([[0, 0], [1, 1], [2, 2]]),
                  key_1=np.array([[0, 0], [1, 1], [2, 2]]),
                  key_2=np.array([[0, 0], [1, 1], [2, 2]]),
              ),
              dict(
                  key_0=np.array([[3, 3], [4, 4], [5, 5]]),
                  key_1=np.array([[3, 3], [4, 4], [5, 5]]),
                  key_2=np.array([[3, 3], [4, 4], [5, 5]]),
              ),
              dict(
                  key_0=np.array([[6, 6], [7, 7], [8, 8]]),
                  key_1=np.array([[6, 6], [7, 7], [8, 8]]),
                  key_2=np.array([[6, 6], [7, 7], [8, 8]]),
              ),
          ],
      ),
  )
  def test_rebatch(
      self, batch_size, rebatch_size, drop_remainder, expected_elements
  ):
    ds = self._get_test_dataset_ten_elements(
        batch_size=batch_size,
        rebatch_size=rebatch_size,
        drop_remainder=drop_remainder,
    )
    actual_elements = list(ds)
    self.assertLen(actual_elements, len(expected_elements))
    self._assert_trees_equal(actual_elements, expected_elements)

  @parameterized.named_parameters(
      dict(
          testcase_name="to_larger_batch",
          batch_size=3,
          rebatch_size=4,
          drop_remainder=False,
          expected_elements=[
              np.array([0, 1, 2, 3]),
              np.array([4, 5, 6, 7]),
              np.array([8, 9]),
          ],
      ),
      dict(
          testcase_name="to_smaller_batch",
          batch_size=4,
          rebatch_size=3,
          drop_remainder=False,
          expected_elements=[
              np.array([0, 1, 2]),
              np.array([3, 4, 5]),
              np.array([6, 7, 8]),
              np.array([9]),
          ],
      ),
      dict(
          testcase_name="to_same_batch",
          batch_size=4,
          rebatch_size=4,
          drop_remainder=False,
          expected_elements=[
              np.array([0, 1, 2, 3]),
              np.array([4, 5, 6, 7]),
              np.array([8, 9]),
          ],
      ),
      dict(
          testcase_name="to_larger_batch_drop_remainder",
          batch_size=3,
          rebatch_size=4,
          drop_remainder=True,
          expected_elements=[
              np.array([0, 1, 2, 3]),
              np.array([4, 5, 6, 7]),
          ],
      ),
      dict(
          testcase_name="to_smaller_batch_drop_remainder",
          batch_size=4,
          rebatch_size=3,
          drop_remainder=True,
          expected_elements=[
              np.array([0, 1, 2]),
              np.array([3, 4, 5]),
              np.array([6, 7, 8]),
          ],
      ),
  )
  def test_rebatch_scalar_elements(
      self, batch_size, rebatch_size, drop_remainder, expected_elements
  ):
    ds = dataset.MapDataset.range(0, 10)
    ds = ds.to_iter_dataset()
    ds = batch.BatchIterDataset(ds, batch_size=batch_size)
    ds = rebatch.RebatchIterDataset(
        ds, batch_size=rebatch_size, drop_remainder=drop_remainder
    )
    actual_elements = list(ds)
    self.assertLen(actual_elements, len(expected_elements))
    self._assert_trees_equal(actual_elements, expected_elements)

  @parameterized.named_parameters(
      dict(
          testcase_name="to_larger_batch",
          batch_size=3,
          rebatch_size=4,
      ),
      dict(
          testcase_name="to_smaller_batch",
          batch_size=4,
          rebatch_size=3,
      ),
      dict(
          testcase_name="to_same_batch",
          batch_size=4,
          rebatch_size=4,
      ),
  )
  def test_rebatch_checkpointing(self, batch_size, rebatch_size):
    ds = self._get_test_dataset_ten_elements(
        batch_size=batch_size, rebatch_size=rebatch_size
    )
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

  def test_empty_batch(self):
    ds = dataset.MapDataset.source(
        [np.full((0, 3), 10), np.full((0, 3), 10), np.full((4, 3), 10)]
    )
    ds = ds.to_iter_dataset()
    ds = rebatch.RebatchIterDataset(ds, batch_size=2)
    ds_iter = iter(ds)
    with self.assertRaises(ValueError):
      next(ds_iter)


if __name__ == "__main__":
  absltest.main()
