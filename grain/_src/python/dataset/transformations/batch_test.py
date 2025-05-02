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

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import batch
from grain._src.python.dataset.transformations import repeat
from grain._src.python.dataset.transformations import source
import numpy as np
import tree


class MakeBatchTest(absltest.TestCase):

  def test_zero_values(self):
    values = []
    with self.assertRaises(ValueError):
      batch._make_batch(values)

  def test_single_value(self):
    values = [np.asarray([1, 2, 3])]
    batched_values = batch._make_batch(values)
    self.assertEqual(batched_values.shape, (1, 3))

  def test_two_values(self):
    values = [np.asarray([1, 2, 3]), np.asarray([4, 5, 6])]
    batched_values = batch._make_batch(values)
    self.assertEqual(batched_values.shape, (2, 3))

  def test_different_shape(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"a": np.asarray([4, 5])}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch._make_batch(values)

  def test_different_structure(self):
    values = [{"a": np.asarray([1, 2, 3])}, {"b": np.asarray(0.5)}]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch._make_batch(values)
    values = [
        {"a": np.asarray([1, 2, 3])},
        {"b": np.asarray(0.5)},
        {"c": np.asarray(True)},
    ]
    with self.assertRaisesRegex(
        ValueError,
        "Expected all input elements to have the same structure but got:",
    ):
      batch._make_batch(values)


class BatchMapDatasetTest(parameterized.TestCase):

  def test_batch_size_2(self):
    ds = dataset.MapDataset.range(0, 10)
    ds = batch.BatchMapDataset(ds, batch_size=2)
    self.assertLen(ds, 5)  # 10 // 2 = 5.
    actual = [ds[i] for i in range(5)]
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    np.testing.assert_allclose(actual, expected)

  def test_custom_batch_fn(self):
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

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_batch_size_3(self, drop_remainder: bool):
    ds = dataset.MapDataset.range(0, 10)
    ds = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=drop_remainder)
    if drop_remainder:
      self.assertLen(ds, 3)  # 10 // 3.
      expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    else:
      self.assertLen(ds, 4)  # ceil(10 / 3).
      expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    actual = [ds[i] for i in range(len(ds))]
    for i in range(len(ds)):
      np.testing.assert_allclose(actual[i], expected[i])

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries(self, drop_remainder: bool):
    num_epochs = 4
    ds = dataset.MapDataset.range(0, 10)
    ds = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=drop_remainder)
    if drop_remainder:
      self.assertLen(ds, 3)
      expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    else:
      self.assertLen(ds, 4)
      expected = num_epochs * [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    actual = [ds[i] for i in range(num_epochs * len(ds))]
    for i in range(len(actual)):
      np.testing.assert_allclose(actual[i], expected[i])

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries_repeat_after_batch(self, drop_remainder: bool):
    num_epochs = 2
    ds = dataset.MapDataset.range(0, 10)
    ds = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=drop_remainder)
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

  @parameterized.named_parameters(
      dict(testcase_name="drop_remainder", drop_remainder=True),
      dict(testcase_name="", drop_remainder=False),
  )
  def test_epoch_boundaries_repeat_before_batch(self, drop_remainder: bool):
    num_epochs = 2
    ds = dataset.MapDataset.range(0, 10)
    ds = repeat.RepeatMapDataset(ds, num_epochs=num_epochs)
    ds = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=drop_remainder)
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

  def test_batch_after_filter_raises_error(self):
    ds = dataset.MapDataset.range(0, 10).filter(lambda x: x % 2 == 0)
    with self.assertRaisesRegex(
        ValueError,
        "`MapDataset.batch` can not follow `MapDataset.filter`",
    ):
      _ = batch.BatchMapDataset(ds, batch_size=3, drop_remainder=True)


class BatchIterDatasetTest(absltest.TestCase):

  def test_batch_size_2(self):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    ds = batch.BatchIterDataset(ds, batch_size=2)
    ds_iter = iter(ds)
    actual = [next(ds_iter) for _ in range(5)]
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    np.testing.assert_allclose(actual, expected)

  def test_custom_batch_fn(self):
    iter_ds = source.SourceMapDataset(
        [{"a": f"element_{i}"} for i in range(10)]
    ).to_iter_dataset()

    def _batch_fn(xs):
      return tree.map_structure(lambda *x: tuple(x), *xs)

    iter_ds = batch.BatchIterDataset(iter_ds, batch_size=2, batch_fn=_batch_fn)
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

  def test_batch_size_3(self):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    # drop_remainder defaults to False
    ds = batch.BatchIterDataset(ds, batch_size=3)
    actual = list(ds)
    self.assertLen(actual, 4)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    for i in range(4):
      np.testing.assert_allclose(actual[i], expected[i])

  def test_batch_size_3_drop_remainder(self):
    ds = dataset.MapDataset.range(0, 10).to_iter_dataset()
    ds = batch.BatchIterDataset(ds, batch_size=3, drop_remainder=True)
    actual = list(ds)
    self.assertLen(actual, 3)
    expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
  absltest.main()
