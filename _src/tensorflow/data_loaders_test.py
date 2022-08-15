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
"""Unit tests for the data_loaders module."""
from typing import Mapping

import grain._src.core.constants as gc
from grain._src.tensorflow import data_loaders
import numpy as np
import tensorflow as tf


def _dict_to_dataset(d: Mapping[str, np.ndarray]) -> tf.data.Dataset:
  return tf.data.Dataset.from_tensor_slices(d)


def _dataset_to_dict(ds: tf.data.Dataset) -> Mapping[str, np.ndarray]:
  assert ds.cardinality() > 0 and ds.cardinality() < 1000
  return next(ds.batch(1000).as_numpy_iterator())


class DataLadersTest(tf.test.TestCase):
  """Tests for the data_loaders module."""

  def test_add_global_record_key(self):
    # Let say we have 3 datasets with 3, 8, 5 records each.
    records_per_dataset = (3, 8, 5)
    num_records = sum(records_per_dataset)
    index = {
        gc.RECORD_KEY:
            np.concatenate((range(3), range(8), range(5)),
                           axis=0),
        gc.DATASET_INDEX:
            np.asarray(3 * [0] + 8 * [1] + 5 * [2], np.int64),
    }
    index_ds = _dict_to_dataset(index)
    actual_ds = data_loaders._add_global_record_key(
        index_ds,
        records_per_dataset=records_per_dataset,
        output_key="my_output_key")
    actual = _dataset_to_dict(actual_ds)
    expected = index | {"my_output_key": np.arange(num_records, dtype=np.int64)}
    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.test.main()
