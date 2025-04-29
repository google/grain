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
from multiprocessing import shared_memory
import threading
import time
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing
from grain._src.python import record
from grain._src.python.operations import BatchOperation
from grain._src.python.shared_memory_array import SharedMemoryArray
from grain._src.python.shared_memory_array import SharedMemoryArrayMetadata
import jax
import numpy as np


def _create_and_delete_shm() -> SharedMemoryArrayMetadata:
  data = np.array([[1, 2], [3, 4]], dtype=np.int32)
  shm_array = SharedMemoryArray(data.shape, data.dtype)
  shm_array.unlink_on_del()
  metadata = shm_array.metadata
  return metadata


def _wait_for_deletion(metadata: SharedMemoryArrayMetadata) -> None:
  while True:
    try:
      _ = shared_memory.SharedMemory(name=metadata.name, create=False)
      time.sleep(0.1)
    except FileNotFoundError:
      break


class SharedMemoryArrayTest(parameterized.TestCase):

  @parameterized.parameters([
      "numpy",
      "jax",
  ])
  def test_batch_dict_of_data_with_shared_memory(self, mode):
    data = [[1, 2], [3, 4]]
    if mode == "numpy":
      data = list(map(lambda x: np.array(x, dtype=np.int32), data))
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
    shm_metadata = actual_record.data["a"]
    self.assertIsInstance(shm_metadata, SharedMemoryArrayMetadata)
    self.assertEqual(shm_metadata.shape, (2, 2))
    self.assertEqual(shm_metadata.dtype, np.int32)
    # Clean up the allocated shared memory block and make sure it no longer
    # exists.
    shm_metadata.close_and_unlink_shm()
    with self.assertRaises(FileNotFoundError):
      _ = shared_memory.SharedMemory(name=shm_metadata.name, create=False)

  def test_async_unlink_limit(self):
    SharedMemoryArray._disable_async_del()
    SharedMemoryArray.enable_async_del(max_outstanding_requests=1)
    event = threading.Event()
    original_close_shm_async = SharedMemoryArray.close_shm_async

    def _wait_for_event(shm, unlink_on_del):
      event.wait(timeout=60)
      original_close_shm_async(shm, unlink_on_del)

    with mock.patch.object(
        SharedMemoryArray, "close_shm_async", side_effect=_wait_for_event
    ):
      metadata = _create_and_delete_shm()
      time.sleep(1)
      # This should succeed, since the unlink request is async and we haven't
      # yet allowed it to progress past the event.
      _ = shared_memory.SharedMemory(name=metadata.name, create=False)

      # All outstanding requests in use, so this should delete the shared memory
      # right away.
      metadata_2 = _create_and_delete_shm()
      with self.assertRaises(FileNotFoundError):
        _ = shared_memory.SharedMemory(name=metadata_2.name, create=False)

      event.set()
      _wait_for_deletion(metadata)

  def test_del_no_pool(self):
    SharedMemoryArray._disable_async_del()
    # Tests deletion of SharedMemory resource when enable_async_del is not
    # called.
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shm_array = SharedMemoryArray(data.shape, data.dtype)
    shm_array.unlink_on_del()
    metadata = shm_array.metadata
    del shm_array
    with self.assertRaises(FileNotFoundError):
      _ = shared_memory.SharedMemory(name=metadata.name, create=False)

  def test_del_many_async(self):
    SharedMemoryArray._disable_async_del()
    SharedMemoryArray.enable_async_del(
        num_threads=4, max_outstanding_requests=20
    )
    shm_metadatas = [_create_and_delete_shm() for _ in range(50)]
    for metadata in shm_metadatas:
      _wait_for_deletion(metadata)

  def test_del_many_async_reuse_pool(self):
    max_outstanding_requests = 20
    SharedMemoryArray._disable_async_del()
    SharedMemoryArray.enable_async_del(
        num_threads=4, max_outstanding_requests=max_outstanding_requests
    )
    original_close_shm_async = SharedMemoryArray.close_shm_async

    def my_close_shm_async(shm, unlink_on_del):
      original_close_shm_async(shm, unlink_on_del)

    with mock.patch.object(
        SharedMemoryArray, "close_shm_async", side_effect=my_close_shm_async
    ) as mock_close_shm_async:
      with self.subTest("first_round_of_requests"):
        shm_metadatas = [
            _create_and_delete_shm() for _ in range(max_outstanding_requests)
        ]
        for metadata in shm_metadatas:
          _wait_for_deletion(metadata)
        self.assertEqual(
            max_outstanding_requests, mock_close_shm_async.call_count
        )
      with self.subTest("second_round_of_requests"):
        # Do it again to make sure the pool is reused.
        shm_metadatas = [
            _create_and_delete_shm() for _ in range(max_outstanding_requests)
        ]
        for metadata in shm_metadatas:
          _wait_for_deletion(metadata)
        self.assertEqual(
            2 * max_outstanding_requests, mock_close_shm_async.call_count
        )


if __name__ == "__main__":
  absltest.main()
