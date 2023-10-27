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
"""Tests for np_array_in_shared_memory."""

from unittest import mock

from absl.testing import absltest
import multiprocessing
from grain._src.python import np_array_in_shared_memory
import numpy as np


class NpArrayInSharedMemoryTest(absltest.TestCase):

  @mock.patch.object(np_array_in_shared_memory, "reduction")
  def test_enable_numpy_shared_memory_pickler(self, reduction_mock):
    forking_pickler_mock = mock.MagicMock()
    reduction_mock.ForkingPickler = forking_pickler_mock
    np_array_in_shared_memory.enable_numpy_shared_memory_pickler()
    forking_pickler_mock.register.assert_called_with(
        np.ndarray, np_array_in_shared_memory._reduce_ndarray
    )

  def test_reduce_np_array_skip(self):
    class DT:
      pass

    arr1 = np.array([127, 128, 129], dtype=np.dtype(DT))
    res = np_array_in_shared_memory._reduce_ndarray(arr1)
    self.assertEqual(res, arr1.__reduce__())  # pytype: disable=attribute-error

    arr2 = np.arange(8, dtype="int8")
    arr2.reshape(2, 4, order="F")
    res = np_array_in_shared_memory._reduce_ndarray(arr2)
    self.assertEqual(res, arr2.__reduce__())  # pytype: disable=attribute-error

    arr3 = np.array([1, 2, 3, 4])
    res = np_array_in_shared_memory._reduce_ndarray(arr3)
    self.assertEqual(res, arr3.__reduce__())  # pytype: disable=attribute-error

  def test_reduce_np_array_no_skip(self):
    arr = np.arange(10.0)
    np_array_in_shared_memory._reduce_ndarray(arr)


if __name__ == "__main__":
  absltest.main()
