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
import pickle
import platform
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing
from grain._src.python import operations
from grain._src.python import record
from grain._src.python.ipc import shared_memory_array
import jax
import numpy as np

SharedMemoryArray = shared_memory_array.SharedMemoryArray
SharedMemoryArrayMetadata = shared_memory_array.SharedMemoryArrayMetadata
copy_to_shm = shared_memory_array.copy_to_shm
open_from_shm = shared_memory_array.open_from_shm
unlink_shm = shared_memory_array.unlink_shm
BatchOperation = operations.BatchOperation


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


@absltest.skipIf(platform.system() == "Windows", "Timeouts on Windows.")
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

    input_data = iter([
        record.Record(
            record.RecordMetadata(index=idx, record_key=idx + 1),
            {"a": item},
        )
        for idx, item in enumerate(data)
    ])

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

  def test_copy_and_open_shm_single_array(self):
    arr = np.arange(10).astype(np.int32)
    shm_struct = copy_to_shm(arr)
    self.assertIsInstance(shm_struct, SharedMemoryArrayMetadata)
    opened_struct = open_from_shm(shm_struct)
    self.assertIsInstance(opened_struct, SharedMemoryArray)
    np.testing.assert_array_equal(opened_struct, arr)
    self.assertTrue(opened_struct._unlink_on_del)

  def test_copy_and_open_shm_nested_structure(self):
    arr = np.arange(10).astype(np.int32)
    arr2 = np.arange(5).astype(np.int32)
    struct = {"a": arr, "b": [arr2, arr], "c": 123}
    shm_struct = copy_to_shm(struct)
    self.assertIsInstance(shm_struct["a"], SharedMemoryArrayMetadata)
    self.assertIsInstance(shm_struct["b"][0], SharedMemoryArrayMetadata)
    self.assertIsInstance(shm_struct["b"][1], SharedMemoryArrayMetadata)
    self.assertEqual(shm_struct["c"], 123)

    opened_struct = open_from_shm(shm_struct)
    self.assertIsInstance(opened_struct["a"], SharedMemoryArray)
    np.testing.assert_array_equal(opened_struct["a"], arr)
    self.assertTrue(opened_struct["a"]._unlink_on_del)
    self.assertIsInstance(opened_struct["b"][0], SharedMemoryArray)
    np.testing.assert_array_equal(opened_struct["b"][0], arr2)
    self.assertTrue(opened_struct["b"][0]._unlink_on_del)
    self.assertIsInstance(opened_struct["b"][1], SharedMemoryArray)
    np.testing.assert_array_equal(opened_struct["b"][1], arr)
    self.assertTrue(opened_struct["b"][1]._unlink_on_del)
    self.assertEqual(opened_struct["c"], 123)

  def test_copy_and_open_shm_min_size(self):
    arr = np.arange(10).astype(np.int32)  # 40 bytes
    shm_struct = copy_to_shm(arr, min_size=100)
    self.assertIsInstance(shm_struct, np.ndarray)
    np.testing.assert_array_equal(shm_struct, arr)
    shm_struct = copy_to_shm(arr, min_size=10)
    self.assertIsInstance(shm_struct, SharedMemoryArrayMetadata)
    opened_struct = open_from_shm(shm_struct)
    self.assertIsInstance(opened_struct, SharedMemoryArray)
    np.testing.assert_array_equal(opened_struct, arr)
    self.assertTrue(opened_struct._unlink_on_del)

  def test_unlink_shm(self):
    arr = np.arange(10).astype(np.int32)
    arr2 = np.arange(5).astype(np.int32)
    struct = {"a": arr, "b": [arr2, arr], "c": 123}
    shm_struct = copy_to_shm(struct)
    names = [
        shm_struct["a"].name,
        shm_struct["b"][0].name,
        shm_struct["b"][1].name,
    ]
    # Check that SHMs exist.
    for name in names:
      self.assertIsNotNone(shared_memory.SharedMemory(name=name, create=False))

    unlink_shm(shm_struct)

    for name in names:
      with self.assertRaises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)

  def test_advanced_indexing_returns_numpy_array(self):
    shm_arr = SharedMemoryArray((10, 2), np.int32)
    shm_arr.unlink_on_del()
    # Slicing returns a view backed by shared memory:
    sliced = shm_arr[0:2]
    self.assertIsInstance(sliced, SharedMemoryArray)
    self.assertIsNotNone(sliced.shm)

    # Advanced indexing returns a regular NumPy array:
    advanced = shm_arr[[0, 2]]
    self.assertNotIsInstance(advanced, SharedMemoryArray)
    self.assertIsInstance(advanced, np.ndarray)

  def test_close_and_unlink_shm_already_unlinked_logs_warning(self):
    shm_meta = copy_to_shm(np.arange(10, dtype=np.int32))
    shm_meta.close_and_unlink_shm()
    with self.assertLogs(level="WARNING") as logs:
      shm_meta.close_and_unlink_shm()
    self.assertTrue(any("was already unlinked" in log for log in logs.output))

  def test_re_copy_opened_shm_allocates_new_block(self):
    SharedMemoryArray._disable_async_del()
    arr = np.arange(10, dtype=np.int32)
    shm_meta_1 = copy_to_shm(arr)
    opened_1 = open_from_shm(shm_meta_1)
    self.assertTrue(opened_1._unlink_on_del)

    shm_meta_2 = copy_to_shm(opened_1)
    self.assertTrue(opened_1._unlink_on_del)
    self.assertNotEqual(shm_meta_1.name, shm_meta_2.name)

    del opened_1
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=shm_meta_1.name, create=False)
    shm_2 = shared_memory.SharedMemory(name=shm_meta_2.name, create=False)
    shm_2.close()

    opened_2 = open_from_shm(shm_meta_2)
    np.testing.assert_array_equal(opened_2, arr)

    del opened_2
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=shm_meta_2.name, create=False)

  def test_copy_to_shm_with_shared_memory_array_slices(self):
    SharedMemoryArray._disable_async_del()
    shm_arr = SharedMemoryArray((8,), dtype=np.int32)
    shm_arr[:] = np.arange(8, dtype=np.int32)
    shm_arr.unlink_on_del()
    orig_name = shm_arr.metadata.name

    shm_meta_dict = copy_to_shm({"first": shm_arr[:4], "second": shm_arr[4:]})
    first_name = shm_meta_dict["first"].name
    second_name = shm_meta_dict["second"].name
    self.assertNotEqual(first_name, orig_name)
    self.assertNotEqual(second_name, orig_name)
    self.assertNotEqual(first_name, second_name)

    del shm_arr
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=orig_name, create=False)

    opened = open_from_shm(shm_meta_dict)
    np.testing.assert_array_equal(opened["first"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(
        opened["second"], np.arange(4, 8, dtype=np.int32)
    )

    del opened
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=first_name, create=False)
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=second_name, create=False)

  def test_aliased_shared_memory_array_in_pytree(self):
    SharedMemoryArray._disable_async_del()
    shm_arr = SharedMemoryArray((4,), dtype=np.int32)
    shm_arr[:] = np.arange(4, dtype=np.int32)
    orig_name = shm_arr.metadata.name

    shm_meta_dict = copy_to_shm({"a": shm_arr, "b": shm_arr})
    self.assertEqual(shm_meta_dict["a"].name, orig_name)
    self.assertEqual(shm_meta_dict["b"].name, orig_name)

    del shm_arr
    shm = shared_memory.SharedMemory(name=orig_name, create=False)
    shm.close()

    opened = open_from_shm(shm_meta_dict)
    np.testing.assert_array_equal(opened["a"], np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(opened["b"], np.arange(4, dtype=np.int32))

    del opened
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=orig_name, create=False)

  def test_copy_to_shm_fallback_converts_shared_memory_array_to_ndarray(self):
    # pylint: disable=g-unsafe-pickle-load
    SharedMemoryArray._disable_async_del()
    shm_arr = SharedMemoryArray((6,), dtype=np.int32)
    shm_arr[:] = np.arange(6, dtype=np.int32)
    shm_arr.unlink_on_del()
    orig_name = shm_arr.metadata.name

    res = copy_to_shm(
        {"strided": shm_arr[::2], "empty": shm_arr[:0], "small": shm_arr},
        min_size=100,
    )
    self.assertIs(type(res["strided"]), np.ndarray)
    self.assertIs(type(res["empty"]), np.ndarray)
    self.assertIs(type(res["small"]), np.ndarray)

    unpickled = pickle.loads(pickle.dumps(res))
    self.assertIs(type(unpickled["strided"]), np.ndarray)
    self.assertIs(type(unpickled["empty"]), np.ndarray)
    self.assertIs(type(unpickled["small"]), np.ndarray)
    np.testing.assert_array_equal(unpickled["strided"], [0, 2, 4])
    np.testing.assert_array_equal(unpickled["empty"], [])
    np.testing.assert_array_equal(unpickled["small"], [0, 1, 2, 3, 4, 5])

    del shm_arr, res, unpickled
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=orig_name, create=False)

  def test_unlink_shm_with_duplicate_metadata(self):
    shm_meta = copy_to_shm(np.arange(10, dtype=np.int32))
    name = shm_meta.name
    shm = shared_memory.SharedMemory(name=name, create=False)
    shm.close()

    with self.assertLogs(level="WARNING") as logs:
      unlink_shm({"a": shm_meta, "b": shm_meta})
    self.assertTrue(any("was already unlinked" in log for log in logs.output))
    with self.assertRaises(FileNotFoundError):
      shared_memory.SharedMemory(name=name, create=False)

  def test_view_is_pickled_by_value(self):
    # pylint: disable=g-unsafe-pickle-load
    SharedMemoryArray._disable_async_del()
    shm_arr = SharedMemoryArray((8,), dtype=np.int32)
    shm_arr[:] = np.arange(8, dtype=np.int32)
    shm_arr.unlink_on_del()

    for view, expected in (
        (shm_arr[4:], np.arange(4, 8, dtype=np.int32)),
        (shm_arr[::2], np.arange(0, 8, 2, dtype=np.int32)),
    ):
      self.assertIsInstance(view, SharedMemoryArray)
      unpickled = pickle.loads(pickle.dumps(view))
      self.assertNotIsInstance(unpickled, SharedMemoryArray)
      np.testing.assert_array_equal(unpickled, expected)

    # A whole array still pickles by reference to the shared memory block.
    unpickled = pickle.loads(pickle.dumps(shm_arr))
    self.assertIsInstance(unpickled, SharedMemoryArray)
    self.assertEqual(unpickled.metadata.name, shm_arr.metadata.name)
    np.testing.assert_array_equal(unpickled, np.arange(8, dtype=np.int32))

  def test_async_del_failure_releases_semaphore(self):
    SharedMemoryArray._disable_async_del()
    SharedMemoryArray.enable_async_del(
        num_threads=1, max_outstanding_requests=1
    )
    semaphore = SharedMemoryArray._outstanding_del_requests
    assert semaphore is not None
    original_del_shm = shared_memory_array._del_shm
    shm_arr = SharedMemoryArray((4,), dtype=np.int32)
    shm_arr.unlink_on_del()
    shm = shm_arr.shm
    assert isinstance(shm, shared_memory.SharedMemory)

    with mock.patch.object(
        shared_memory_array, "_del_shm", side_effect=RuntimeError("boom")
    ) as mock_del_shm:
      del shm_arr
      deadline = time.time() + 30
      while mock_del_shm.call_count == 0 and time.time() < deadline:
        time.sleep(0.01)
      self.assertEqual(mock_del_shm.call_count, 1)
      deadline = time.time() + 30
      while not semaphore.acquire(blocking=False):
        self.assertLess(time.time(), deadline, "semaphore slot leaked")
        time.sleep(0.01)
      semaphore.release()
    original_del_shm(shm, unlink=True)

  def test_zero_itemsize_dtype_not_copied_to_shm(self):
    arr = np.zeros((4,), dtype="V0")
    res = copy_to_shm(arr)
    self.assertIs(res, arr)


if __name__ == "__main__":
  absltest.main()
