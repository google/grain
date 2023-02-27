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
"""Test for BatchAndPackDataset."""

from absl.testing import parameterized
from grain._src.tensorflow.ops import merge_by_min_value
import numpy as np
import tensorflow as tf


class BatchAndPackDatasetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the BatchAndPack dataset."""

  def test_one_dataset(self):
    ds = tf.data.experimental.from_list([
        {
            "index": np.asarray(0, np.int64),
            "data": 1.23,
        },
        {
            "index": np.asarray(2, np.int64),
            "data": 2.23,
        },
        {
            "index": np.asarray(4, np.int64),
            "data": 3.23,
        },
    ])
    ds = merge_by_min_value.MergeByMinValueDataset([ds], merge_field="index")
    self.assertEqual(ds.cardinality(), 3)
    ds = list(ds.as_numpy_iterator())
    self.assertAllClose(
        [
            {"index": np.asarray(0, np.int64), "data": 1.23},
            {"index": np.asarray(2, np.int64), "data": 2.23},
            {"index": np.asarray(4, np.int64), "data": 3.23},
        ],
        ds,
    )

  def test_two_datasets(self):
    ds1 = tf.data.experimental.from_list([
        {
            "index": np.asarray(0, np.int64),
            "data": 1.23,
        },
        {
            "index": np.asarray(2, np.int64),
            "data": 2.23,
        },
        {
            "index": np.asarray(4, np.int64),
            "data": 3.23,
        },
    ])
    ds2 = tf.data.experimental.from_list([
        {
            "index": np.asarray(1, np.int64),
            "data": 4.23,
        },
        {
            "index": np.asarray(5, np.int64),
            "data": 5.23,
        },
        {
            "index": np.asarray(6, np.int64),
            "data": 6.23,
        },
    ])
    ds = merge_by_min_value.MergeByMinValueDataset(
        [ds1, ds2], merge_field="index"
    )
    self.assertEqual(ds.cardinality(), 6)
    ds = list(ds.as_numpy_iterator())
    self.assertAllClose(
        [
            {"index": np.asarray(0, np.int64), "data": 1.23},
            {"index": np.asarray(1, np.int64), "data": 4.23},
            {"index": np.asarray(2, np.int64), "data": 2.23},
            {"index": np.asarray(4, np.int64), "data": 3.23},
            {"index": np.asarray(5, np.int64), "data": 5.23},
            {"index": np.asarray(6, np.int64), "data": 6.23},
        ],
        ds,
    )

  def test_three_datasets(self):
    ds1 = tf.data.experimental.from_list([
        {
            "index": np.asarray(0, np.int64),
            "data": 1.23,
        },
        {
            "index": np.asarray(2, np.int64),
            "data": 2.23,
        },
    ])
    ds2 = tf.data.experimental.from_list([
        {
            "index": np.asarray(1, np.int64),
            "data": 4.23,
        },
        {
            "index": np.asarray(5, np.int64),
            "data": 5.23,
        },
        {
            "index": np.asarray(7, np.int64),
            "data": 6.23,
        },
    ])
    ds3 = tf.data.experimental.from_list([
        {
            "index": np.asarray(6, np.int64),
            "data": 7.23,
        },
        {
            "index": np.asarray(8, np.int64),
            "data": 8.23,
        },
    ])
    ds = merge_by_min_value.MergeByMinValueDataset(
        [ds1, ds2, ds3], merge_field="index"
    )
    self.assertEqual(ds.cardinality(), 7)
    ds = list(ds.as_numpy_iterator())
    self.assertAllClose(
        [
            {"index": np.asarray(0, np.int64), "data": 1.23},
            {"index": np.asarray(1, np.int64), "data": 4.23},
            {"index": np.asarray(2, np.int64), "data": 2.23},
            {"index": np.asarray(5, np.int64), "data": 5.23},
            {"index": np.asarray(6, np.int64), "data": 7.23},
            {"index": np.asarray(7, np.int64), "data": 6.23},
            {"index": np.asarray(8, np.int64), "data": 8.23},
        ],
        ds,
    )

  def test_inifinte_cardinality(self):
    ds1 = tf.data.experimental.from_list([
        {
            "index": np.asarray(0, np.int64),
            "data": 1.23,
        },
    ]).repeat()
    ds2 = tf.data.experimental.from_list([
        {
            "index": np.asarray(1, np.int64),
            "data": 4.23,
        },
        {
            "index": np.asarray(5, np.int64),
            "data": 5.23,
        },
    ])
    ds = merge_by_min_value.MergeByMinValueDataset(
        [ds1, ds2], merge_field="index"
    )
    self.assertEqual(ds.cardinality(), tf.data.INFINITE_CARDINALITY)


if __name__ == "__main__":
  tf.test.main()
