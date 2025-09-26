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

import functools
import importlib
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import batch
from grain._src.python.dataset.transformations import repeat
from grain._src.python.dataset.transformations import source
import numpy as np
import tree


class MakeBatchTest(absltest.TestCase):

  def test_batch_zero_values_error(self):
    values = []
    with self.assertRaises(ValueError):
      batch.make_batch(values)

  def test_batch_single_value_success(self):
    values = [np.asarray([1, 2, 3])]
    batched_values = batch.make_batch(values)
    self.assertEqual(batched_values.shape, (1, 3))

  def test_batch_single_value_parallel_batch_enabled_success(self):
    values = [np.asarray([1, 2, 3])]
    make_batch_parallel = batch._MakeBatchParallel()
    batched_values = make_batch_parallel(values)
    self.assertEqual(batched_values.shape, (1, 3))

  def test_batch_two_values_success(self):
    values = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    batched_values = batch.make_batch(values)
    self.assertEqual(batched_values.shape, (2, 3))

  def test_batch_two_values_parallel_batch_enabled_success(self):
    values = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    make_batch_parallel = batch._MakeBatchParallel()
    batched_values = make_batch_parallel(values)
    self.assertEqual(batched_values.shape, (2, 3))

  def test_batch_different_shapes_error(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"a": np.asarray([4, 5])}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)

  def test_batch_different_shapes_parallel_batch_enabled_error(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"a": np.asarray([4, 5])}]
    make_batch_parallel = batch._MakeBatchParallel()
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      make_batch_parallel(values)

  def test_batch_different_structures_error(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"b": np.asarray(0.5)}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)
    values = [
        {"a": np.asarray([1, 2, 3])},
        {"b": np.asarray(0.5)},
        {"c": np.asarray(True)},
    ]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)

  def test_batch_different_structures_parallel_batch_enabled_error(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"b": np.asarray(0.5)}]
    make_batch_parallel = batch._MakeBatchParallel()
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      make_batch_parallel(values)
      values = [
          {"a": np.asarray([1, 2, 3])},
          {"b": np.asarray(0.5)},
          {"c": np.asarray(True)},
      ]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      make_batch_parallel(values)


class BatchAndPadTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="scalar_values_no_padding",
          values=[1, 2, 3],
          batch_size=3,
          pad_value=0,
          expected=np.array([1, 2, 3]),
      ),
      dict(
          testcase_name="scalar_values_with_padding",
          values=[1, 2, 3],
          batch_size=5,
          pad_value=1,
          expected=np.array([1, 2, 3, 1, 1]),
      ),
      dict(
          testcase_name="dicts_with_padding",
          values=[
              {
                  "a": np.array([1, 2, 3]),
                  "b": np.array([[1.2], [0.2]], dtype=np.float32),
              },
              {
                  "a": np.array([4, 5, 6]),
                  "b": np.array([[323.1], [2222.3]], dtype=np.float32),
              },
          ],
          batch_size=4,
          pad_value=0,
          expected={
              "a": np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]]),
              "b": np.array(
                  [
                      [[1.2], [0.2]],
                      [[323.1], [2222.3]],
                      [[0.0], [0.0]],
                      [[0.0], [0.0]],
                  ],
                  dtype=np.float32,
              ),
          },
      ),
      dict(
          testcase_name="nested_dicts_with_padding",
          values=[
              {
                  "aa": {"a": np.array([1, 2, 3])},
                  "b": np.array([[1.2], [0.2]], dtype=np.float32),
              },
              {
                  "aa": {"a": np.array([4, 5, 6])},
                  "b": np.array([[323.1], [2222.3]], dtype=np.float32),
              },
          ],
          batch_size=4,
          pad_value=0,
          expected={
              "aa": {
                  "a": np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0]])
              },
              "b": np.array(
                  [
                      [[1.2], [0.2]],
                      [[323.1], [2222.3]],
                      [[0.0], [0.0]],
                      [[0.0], [0.0]],
                  ],
                  dtype=np.float32,
              ),
          },
      ),
  )
  def test_correct_output(self, values, batch_size, pad_value, expected):
    np.testing.assert_equal(
        batch.batch_and_pad(values, batch_size=batch_size, pad_value=pad_value),
        expected,
    )

  def test_zero_values(self):
    values = []
    with self.assertRaises(ValueError):
      batch.batch_and_pad(values, batch_size=1)

  def test_different_shape(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"a": np.asarray([4, 5])}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)

  def test_different_structure(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"b": np.asarray(0.5)}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)
    values = [
        {"a": np.asarray([1, 2, 3])},
        {"b": np.asarray(0.5)},
        {"c": np.asarray(True)},
    ]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch.make_batch(values)


class BatchMapDatasetTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    importlib.reload(batch.tree_lib)
    importlib.reload(batch)

  @parameterized.named_parameters(
      dict(
          testcase_name="range_ds_jax",
          use_jax=True,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
      ),
      dict(
          testcase_name="range_ds_no_jax",
          use_jax=False,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
      ),
      dict(
          testcase_name="source_ds_jax",
          use_jax=True,
          initial_ds=source.SourceMapDataset([
              np.asarray([1, 2, 3]),
              np.asarray([4, 5, 6]),
              np.asarray([7, 8, 9]),
              np.asarray([10, 11, 12]),
          ]),
          expected=[
              [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])],
              [np.asarray([7, 8, 9]), np.asarray([10, 11, 12])],
          ],
      ),
      dict(
          testcase_name="source_ds_no_jax",
          use_jax=False,
          initial_ds=source.SourceMapDataset([
              np.asarray([1, 2, 3]),
              np.asarray([4, 5, 6]),
              np.asarray([7, 8, 9]),
              np.asarray([10, 11, 12]),
          ]),
          expected=[
              [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])],
              [np.asarray([7, 8, 9]), np.asarray([10, 11, 12])],
          ],
      ),
  )
  def test_batch_size_2(self, use_jax: bool, initial_ds, expected):
    def test_batch_size_2_actual_equals_expected(
        expect_parallel_batch: bool = False,
    ):
      ds = batch.BatchMapDataset(initial_ds, batch_size=2)
      if expect_parallel_batch:
        self.assertIsInstance(ds._batch_fn, batch._MakeBatchParallel)
      else:
        self.assertIs(ds._batch_fn, batch.make_batch)
      self.assertLen(ds, len(expected))  # 10 // 2 = 5.
      actual = [ds[i] for i in range(len(ds))]
      np.testing.assert_allclose(actual, expected)

    with mock.patch.dict(
        sys.modules, {"jax": sys.modules["jax"] if use_jax else None}
    ):
      importlib.reload(batch.tree_lib)
      test_batch_size_2_actual_equals_expected()

  def test_custom_batch_fn(self):

    def custom_batch_fn_actual_equals_expected():
      ds = source.SourceMapDataset([{"a": f"element_{i}"} for i in range(10)])

      def _batch_fn(xs):
        return tree.map_structure(lambda *x: tuple(x), *xs)

      ds = batch.BatchMapDataset(ds, batch_size=2, batch_fn=_batch_fn)
      self.assertLen(ds, 5)  # 10 // 2 = 5.
      actual = [ds[i] for i in range(5)]
      expected = [
          {"a": ("element_0", "element_1")},
          {"a": ("element_2", "element_3")},
          {"a": ("element_4", "element_5")},
          {"a": ("element_6", "element_7")},
          {"a": ("element_8", "element_9")},
      ]
      self.assertEqual(actual, expected)

    custom_batch_fn_actual_equals_expected()

  @parameterized.named_parameters(
      dict(
          testcase_name="range_ds_drop_remainder",
          drop_remainder=True,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
      ),
      dict(
          testcase_name="range_ds_no_drop_remainder",
          drop_remainder=False,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
      ),
      dict(
          testcase_name="source_ds_drop_remainder",
          drop_remainder=True,
          initial_ds=source.SourceMapDataset([
              np.asarray([0, 1, 2]),
              np.asarray([3, 4, 5]),
              np.asarray([6, 7, 8]),
              np.asarray([9, 10, 11]),
          ]),
          expected=[
              [
                  np.asarray([0, 1, 2]),
                  np.asarray([3, 4, 5]),
                  np.asarray([6, 7, 8]),
              ],
          ],
      ),
      dict(
          testcase_name="source_ds_no_drop_remainder",
          drop_remainder=False,
          initial_ds=source.SourceMapDataset([
              np.asarray([0, 1, 2]),
              np.asarray([3, 4, 5]),
              np.asarray([6, 7, 8]),
              np.asarray([9, 10, 11]),
          ]),
          expected=[
              [
                  np.asarray([0, 1, 2]),
                  np.asarray([3, 4, 5]),
                  np.asarray([6, 7, 8]),
              ],
              [np.asarray([9, 10, 11])],
          ],
      ),
  )
  def test_batch_size_3(self, drop_remainder: bool, initial_ds, expected):

    def test_batch_size_3_actual_equals_expected(
        expect_parallel_batch: bool = False,
    ):
      ds = batch.BatchMapDataset(
          initial_ds, batch_size=3, drop_remainder=drop_remainder
      )
      if expect_parallel_batch:
        self.assertIsInstance(ds._batch_fn, batch._MakeBatchParallel)
      else:
        self.assertIs(ds._batch_fn, batch.make_batch)
      self.assertLen(ds, len(expected))
      actual = [ds[i] for i in range(len(ds))]
      for i in range(len(ds)):
        np.testing.assert_allclose(actual[i], expected[i])

    test_batch_size_3_actual_equals_expected()

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries(self, drop_remainder: bool):

    def test_epoch_boundaries_actual_equals_expected():
      num_epochs = 4
      ds = dataset.MapDataset.range(0, 10)
      ds = batch.BatchMapDataset(
          ds, batch_size=3, drop_remainder=drop_remainder
      )
      if drop_remainder:
        self.assertLen(ds, 3)
        expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      else:
        self.assertLen(ds, 4)
        expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
      actual = [ds[i] for i in range(num_epochs * len(ds))]
      for i in range(len(actual)):
        np.testing.assert_allclose(actual[i], expected[i])

    test_epoch_boundaries_actual_equals_expected()

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries_repeat_after_batch(self, drop_remainder: bool):

    def test_epoch_boundaries_repeat_after_batch_actual_equals_expected():
      num_epochs = 2
      ds = dataset.MapDataset.range(0, 10)
      ds = batch.BatchMapDataset(
          ds, batch_size=3, drop_remainder=drop_remainder
      )
      ds = repeat.RepeatMapDataset(ds, num_epochs=num_epochs)
      if drop_remainder:
        self.assertLen(ds, 6)
        # Remainder gets dropped in both epochs.
        expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      else:
        self.assertLen(ds, 8)
        expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
      actual = [ds[i] for i in range(len(ds))]
      for i in range(len(actual)):
        np.testing.assert_allclose(actual[i], expected[i])

    test_epoch_boundaries_repeat_after_batch_actual_equals_expected()

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries_repeat_before_batch(self, drop_remainder: bool):

    def test_epoch_boundaries_repeat_before_batch_actual_equals_expected():
      num_epochs = 2
      ds = dataset.MapDataset.range(0, 10)
      ds = repeat.RepeatMapDataset(ds, num_epochs=num_epochs)
      ds = batch.BatchMapDataset(
          ds, batch_size=3, drop_remainder=drop_remainder
      )
      if drop_remainder:
        self.assertLen(ds, 6)
        # Remainder of last epoch gets dropped.
        expected = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 1],
            [2, 3, 4],
            [5, 6, 7],
        ]
      else:
        self.assertLen(ds, 7)
        expected = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 1],
            [2, 3, 4],
            [5, 6, 7],
            [8, 9],
        ]
      actual = [ds[i] for i in range(len(ds))]
      for i in range(len(actual)):
        np.testing.assert_allclose(actual[i], expected[i])

    test_epoch_boundaries_repeat_before_batch_actual_equals_expected()

  def test_batch_after_filter_raises_error(self):
    ds = dataset.MapDataset.range(0, 10).filter(lambda x: x % 2 == 0)
    with self.assertRaisesRegex(
        ValueError,
        "`MapDataset.batch` can not follow `MapDataset.filter`",
    ):
      _ = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=True)

  def test_batch_with_padding(self):
    ds = dataset.MapDataset.range(1, 10).map(lambda x: {"x": x})
    batch_size = 4
    batch_fn = functools.partial(batch.batch_and_pad, batch_size=batch_size)
    ds = batch.BatchMapDataset(ds, batch_size=batch_size, batch_fn=batch_fn)
    self.assertLen(ds, 3)
    np.testing.assert_equal(
        list(ds),
        [
            {"x": np.array([1, 2, 3, 4])},
            {"x": np.array([5, 6, 7, 8])},
            {"x": np.array([9, 0, 0, 0])},
        ],
    )


class BatchIterDatasetTest(parameterized.TestCase):

  def tearDown(self):
    super().tearDown()
    importlib.reload(batch.tree_lib)
    importlib.reload(batch)

  @parameterized.named_parameters(
      dict(
          testcase_name="range_ds_jax",
          use_jax=True,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
      ),
      dict(
          testcase_name="range_ds_no_jax",
          use_jax=False,
          initial_ds=dataset.MapDataset.range(0, 10),
          expected=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
      ),
      dict(
          testcase_name="source_ds_jax",
          use_jax=True,
          initial_ds=source.SourceMapDataset([
              np.asarray([1, 2, 3]),
              np.asarray([4, 5, 6]),
              np.asarray([7, 8, 9]),
              np.asarray([10, 11, 12]),
          ]),
          expected=[
              [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])],
              [np.asarray([7, 8, 9]), np.asarray([10, 11, 12])],
          ],
      ),
      dict(
          testcase_name="source_ds_no_jax",
          use_jax=False,
          initial_ds=source.SourceMapDataset([
              np.asarray([1, 2, 3]),
              np.asarray([4, 5, 6]),
              np.asarray([7, 8, 9]),
              np.asarray([10, 11, 12]),
          ]),
          expected=[
              [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])],
              [np.asarray([7, 8, 9]), np.asarray([10, 11, 12])],
          ],
      ),
  )
  def test_batch_size_2(self, use_jax: bool, initial_ds, expected):
    def test_batch_size_2_actual_equals_expected(
        expect_parallel_batch: bool = False,
    ):
      ds = initial_ds.to_iter_dataset()
      ds = batch.BatchIterDataset(ds, batch_size=2)
      if expect_parallel_batch:
        self.assertIsInstance(ds._batch_fn, batch._MakeBatchParallel)
      else:
        self.assertIs(ds._batch_fn, batch.make_batch)
      ds_iter = iter(ds)
      actual = [next(ds_iter) for _ in range(len(expected))]
      np.testing.assert_allclose(actual, expected)

    with mock.patch.dict(
        sys.modules, {"jax": sys.modules["jax"] if use_jax else None}
    ):
      importlib.reload(batch.tree_lib)
      test_batch_size_2_actual_equals_expected()

  def test_custom_batch_fn(self):

    def test_custom_batch_fn_actual_equals_expected():
      iter_ds = source.SourceMapDataset(
          [{"a": f"element_{i}"} for i in range(10)]
      ).to_iter_dataset()

      def _batch_fn(xs):
        return tree.map_structure(lambda *x: tuple(x), *xs)

      iter_ds = batch.BatchIterDataset(
          iter_ds, batch_size=2, batch_fn=_batch_fn
      )
      is_iter = iter(iter_ds)
      actual = [next(is_iter) for _ in range(5)]
      expected = [
          {"a": ("element_0", "element_1")},
          {"a": ("element_2", "element_3")},
          {"a": ("element_4", "element_5")},
          {"a": ("element_6", "element_7")},
          {"a": ("element_8", "element_9")},
      ]
      self.assertEqual(actual, expected)

    test_custom_batch_fn_actual_equals_expected()

  @parameterized.named_parameters(
      dict(testcase_name="jax", use_jax=True),
      dict(testcase_name="no_jax", use_jax=False),
  )
  def test_batch_size_3(self, use_jax: bool):
    def test_batch_size_3_actual_equals_expected():
      ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
      # drop_remainder defaults to False
      ds = batch.BatchIterDataset(ds, batch_size=3)
      actual = list(ds)
      self.assertLen(actual, 4)
      expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
      for i in range(4):
        np.testing.assert_allclose(actual[i], expected[i])

    with mock.patch.dict(
        sys.modules, {"jax": sys.modules["jax"] if use_jax else None}
    ):
      importlib.reload(batch.tree_lib)
      test_batch_size_3_actual_equals_expected()

  @parameterized.named_parameters(
      dict(testcase_name="jax", use_jax=True),
      dict(testcase_name="no_jax", use_jax=False),
  )
  def test_batch_size_3_drop_remainder(self, use_jax: bool):
    def test_batch_size_3_drop_remainder_actual_equals_expected():
      ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
      ds = batch.BatchIterDataset(ds, batch_size=3, drop_remainder=True)
      actual = list(ds)
      self.assertLen(actual, 3)
      expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      np.testing.assert_allclose(actual, expected)

    with mock.patch.dict(
        sys.modules, {"jax": sys.modules["jax"] if use_jax else None}
    ):
      importlib.reload(batch.tree_lib)
      test_batch_size_3_drop_remainder_actual_equals_expected()

  def test_batch_with_padding(self):
    ds = (
        dataset.MapDataset.range(1, 10)
        .map(lambda x: {"x": x})
        .to_iter_dataset()
    )
    batch_size = 4
    batch_fn = functools.partial(batch.batch_and_pad, batch_size=batch_size)
    ds = batch.BatchIterDataset(ds, batch_size=batch_size, batch_fn=batch_fn)
    np.testing.assert_equal(
        list(ds),
        [
            {"x": np.array([1, 2, 3, 4])},
            {"x": np.array([5, 6, 7, 8])},
            {"x": np.array([9, 0, 0, 0])},
        ],
    )


if __name__ == "__main__":
  absltest.main()
