# Copyright 2022 Google LLC
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
"""Unit tests for the batching module."""
from typing import Mapping
from unittest import mock

from absl.testing import parameterized
from grain._src.tensorflow import batching
from jax.experimental import multihost_utils
import numpy as np
import tensorflow as tf


def _dataset_to_dict(ds: tf.data.Dataset) -> Mapping[str, np.ndarray]:
  assert ds.cardinality() > 0 and ds.cardinality() < 1000
  return next(ds.batch(1000).as_numpy_iterator())


class BatchingTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the batching module."""

  def test_batch_none(self):
    ds = tf.data.Dataset.range(5)
    self.assertEqual(ds.cardinality(), 5)
    ds = batching.TfBatchNone()(ds)
    elements = list(ds)
    self.assertLen(elements, 5)
    self.assertAllClose(elements, [0, 1, 2, 3, 4])

  @parameterized.parameters(1, 2, tf.data.AUTOTUNE)
  def test_batch(self, num_parallel_calls: int):
    ds = tf.data.Dataset.range(11)
    batch_fn = batching.TfBatch(
        2, drop_remainder=False, num_parallel_calls=num_parallel_calls)
    ds = batch_fn(ds)
    elements = list(ds)
    self.assertLen(elements, 6)
    self.assertAllClose(elements, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9),
                                   (10,)])

  @parameterized.parameters(1, 2, tf.data.AUTOTUNE)
  def test_batch_drop_remainder(self, num_parallel_calls: int):
    ds = tf.data.Dataset.range(11)
    batch_fn = batching.TfBatch(
        2, drop_remainder=True, num_parallel_calls=num_parallel_calls)
    ds = batch_fn(ds)
    elements = list(ds)
    self.assertLen(elements, 5)
    self.assertAllClose(elements, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])

  @mock.patch.object(
      multihost_utils,
      "process_allgather",
      autospec=True,
      return_value=np.asarray([5, 7]))
  def test_batch_with_pad_elements_process_0(self, mock_process_allgather):
    ds = tf.data.Dataset.range(5).map(lambda i: {"index": i})
    batch_fn = batching.TfBatchWithPadElements(2, mask_key="mask")
    ds = batch_fn(ds)
    elements = _dataset_to_dict(ds)
    self.assertAllClose(
        elements, {
            "index": [(0, 1), (2, 3), (4, 0), (0, 0)],
            "mask": [(True, True), (True, True), (True, False), (False, False)],
        })
    mock_process_allgather.assert_called_once()

  @mock.patch.object(
      multihost_utils,
      "process_allgather",
      autospec=True,
      return_value=np.asarray([5, 7]))
  def test_batch_with_pad_elements_process_1(self, mock_process_allgather):
    ds = tf.data.Dataset.range(7).map(lambda i: {"index": i})
    batch_fn = batching.TfBatchWithPadElements(2, mask_key="mask")
    ds = batch_fn(ds)
    elements = _dataset_to_dict(ds)
    self.assertAllClose(
        elements, {
            "index": [(0, 1), (2, 3), (4, 5), (6, 0)],
            "mask": [(True, True), (True, True), (True, True), (True, False)],
        })
    mock_process_allgather.assert_called_once()

  def test_batch_with_pad_element_unknown_cardinality(self):
    ds = tf.data.Dataset.range(7).filter(lambda i: i % 2 == 0)
    self.assertEqual(ds.cardinality(), tf.data.UNKNOWN_CARDINALITY)
    batch_fn = batching.TfBatchWithPadElements(2, mask_key="mask")
    with self.assertRaisesRegex(
        ValueError, r"Dataset has unknown cardinality before batching.+"):
      batch_fn(ds)


if __name__ == "__main__":
  tf.test.main()
