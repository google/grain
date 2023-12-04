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
"""Tests for multiprocessing common functions."""

import multiprocessing
from multiprocessing import pool
import queue

from absl.testing import absltest
from grain._src.python import multiprocessing_common


class MultiProcessingCommonTest(absltest.TestCase):

  def test_add_element_to_queue(self):
    test_queue = multiprocessing.Queue()
    element = 1
    termination_event = multiprocessing.Event()
    self.assertTrue(
        multiprocessing_common.add_element_to_queue(  # pytype: disable=wrong-arg-types
            element=element,
            elements_queue=test_queue,
            should_stop=termination_event.is_set,
        )
    )
    self.assertEqual(test_queue.get(), 1)

  def test_add_element_to_queue_already_terminated(self):
    test_queue = multiprocessing.Queue()
    element = 1
    termination_event = multiprocessing.Event()
    termination_event.set()
    self.assertFalse(
        multiprocessing_common.add_element_to_queue(  # pytype: disable=wrong-arg-types
            element=element,
            elements_queue=test_queue,
            should_stop=termination_event.is_set,
        )
    )
    with self.assertRaises(queue.Empty):
      test_queue.get(timeout=0.1)

  def test_get_element_from_queue(self):
    test_queue = multiprocessing.Queue()
    expected_element = 1
    test_queue.put(expected_element)
    termination_event = multiprocessing.Event()
    actual_element = multiprocessing_common.get_element_from_queue(  # pytype: disable=wrong-arg-types
        elements_queue=test_queue,
        should_stop=termination_event.is_set,
    )
    self.assertEqual(actual_element, expected_element)

  def test_get_element_from_queue_already_terminated(self):
    test_queue = multiprocessing.Queue()
    expected_element = 1
    test_queue.put(expected_element)
    termination_event = multiprocessing.Event()
    termination_event.set()
    actual_element = multiprocessing_common.get_element_from_queue(  # pytype: disable=wrong-arg-types
        elements_queue=test_queue,
        should_stop=termination_event.is_set,
    )
    self.assertEqual(actual_element, multiprocessing_common.SYSTEM_TERMINATED)

  def test_get_async_result(self):
    thread_pool = pool.ThreadPool(1)
    async_result = thread_pool.apply_async(func=lambda x: x + 1, args=(1,))
    termination_event = multiprocessing.Event()
    result = multiprocessing_common.get_async_result(
        should_stop=termination_event.is_set, async_result=async_result
    )
    self.assertEqual(result, 2)

  def test_get_async_result_already_terminated(self):
    thread_pool = pool.ThreadPool(1)
    async_result = thread_pool.apply_async(func=lambda x: x + 1, args=(1,))
    termination_event = multiprocessing.Event()
    termination_event.set()
    result = multiprocessing_common.get_async_result(
        should_stop=termination_event.is_set, async_result=async_result
    )
    self.assertEqual(result, multiprocessing_common.SYSTEM_TERMINATED)


if __name__ == "__main__":
  absltest.main()
