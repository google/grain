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
"""Tests for shared memory array."""
from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing
from grain._src.python import record
from grain._src.python.operations import BatchOperation
from grain._src.python.shared_memory_array import SharedMemoryArray
from grain._src.python.shared_memory_array import SharedMemoryArrayMetadata
import jax
import numpy as np
import tensorflow as tf


class SharedMemoryArrayTest(parameterized.TestCase):

  @parameterized.parameters(["numpy", "tensorflow", "jax"])
  def test_batch_dict_of_data_with_shared_memory(self, mode):
    data = [[1, 2], [3, 4]]
    if mode == "numpy":
      data = list(map(lambda x: np.array(x, dtype=np.int32), data))
    elif mode == "tensorflow":
      data = list(map(tf.constant, data))
    else:
      data = list(map(jax.numpy.array, data))

    input_data = iter(
        [
            record.Record(
                record.RecordMetadata(index=idx, record_key=idx + 1),
                {"a": item},
            )
            for idx, item in enumerate(data)
        ]
    )

    batch_operation = BatchOperation(batch_size=2)
    batch_operation._enable_shared_memory()
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            {"a": SharedMemoryArray((2, 2), np.int32)},
        )
    ]
    expected_record = expected_output_data[0]
    actual_output_data = list(batch_operation(input_data))
    self.assertLen(actual_output_data, len(expected_output_data))
    actual_record = actual_output_data[0]
    # SharedMemory name is determined by OS and not known in advance. Thus
    # checking for values of individual fields.
    self.assertEqual(actual_record.metadata, expected_record.metadata)
    self.assertIsInstance(actual_record.data, dict)
    self.assertEqual(actual_record.data.keys(), {"a"})
    self.assertIsInstance(actual_record.data["a"], SharedMemoryArrayMetadata)
    self.assertEqual(actual_record.data["a"].shape, (2, 2))
    self.assertEqual(actual_record.data["a"].dtype, np.int32)


if __name__ == "__main__":
  absltest.main()
