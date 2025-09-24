# Copyright 2025 Google LLC
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
"""Tests for variable size queue implementations."""

import multiprocessing
import queue
import threading
import time

from absl.testing import absltest
from grain._src.python import variable_size_queue


class VariableSizeQueueTest(absltest.TestCase):

  def test_put_and_get(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    self.assertEqual(q.qsize(), 0)
    q.put(1)
    self.assertEqual(q.qsize(), 1)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.qsize(), 0)

  def test_put_non_blocking_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)

  def test_put_blocking_with_timeout_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=True, timeout=0.1)

  def test_set_max_size_to_increase_capacity(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)
    q.set_max_size(2)
    q.put(2)  # Should not raise.
    self.assertEqual(q.qsize(), 2)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)

  def test_set_max_size_to_decrease_capacity(self):
    q = variable_size_queue.VariableSizeQueue(max_size=2)
    q.put(1)
    q.put(2)
    self.assertEqual(q.qsize(), 2)
    q.set_max_size(1)
    # qsize is 2, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.qsize(), 1)
    # qsize is 1, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 2)
    self.assertEqual(q.qsize(), 0)
    # qsize is 0, max_size is 1. put should succeed.
    q.put(3)
    self.assertEqual(q.qsize(), 1)
    self.assertEqual(q.get(), 3)

  def test_put_blocks_until_item_is_retrieved(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    q.put(1)
    result = []

    def consumer():
      time.sleep(0.1)
      result.append(q.get())

    t = threading.Thread(target=consumer)
    t.start()
    q.put(2)  # This should block until consumer gets item 1.
    self.assertEqual(q.qsize(), 1)
    self.assertEqual(q.get(), 2)
    t.join()
    self.assertEqual(result, [1])

  def test_put_blocks_until_max_size_increases(self):
    q = variable_size_queue.VariableSizeQueue(max_size=1)
    q.put(1)

    def increase_max_size():
      time.sleep(0.1)
      q.set_max_size(2)

    t = threading.Thread(target=increase_max_size)
    t.start()
    q.put(2)  # This should block until max_size is increased.
    self.assertEqual(q.qsize(), 2)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)
    t.join()


class VariableSizeMultiprocessingQueueTest(absltest.TestCase):

  def test_put_and_get(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    self.assertEqual(q.qsize(), 0)
    q.put(1)
    self.assertEqual(q.qsize(), 1)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.qsize(), 0)

  def test_put_non_blocking_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)

  def test_put_blocking_with_timeout_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=True, timeout=0.1)

  def test_set_max_size_to_increase_capacity(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)
    q.set_max_size(2)
    q.put(2)  # Should not raise.
    self.assertEqual(q.qsize(), 2)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)

  def test_set_max_size_to_decrease_capacity(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        2, ctx=multiprocessing.get_context("fork")
    )
    q.put(1)
    q.put(2)
    self.assertEqual(q.qsize(), 2)
    q.set_max_size(1)
    # qsize is 2, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.qsize(), 1)
    # qsize is 1, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 2)
    self.assertEqual(q.qsize(), 0)
    # qsize is 0, max_size is 1. put should succeed.
    q.put(3)
    self.assertEqual(q.qsize(), 1)
    self.assertEqual(q.get(), 3)

  def test_put_blocks_until_item_is_retrieved_from_process(self):
    ctx = multiprocessing.get_context("fork")
    q = variable_size_queue.VariableSizeMultiprocessingQueue(1, ctx=ctx)
    q.put(1)

    with ctx.Manager() as manager:
      result_list = manager.list()

      def consumer(q, result):
        time.sleep(0.1)
        result.append(q.get())

      p = ctx.Process(target=consumer, args=(q, result_list))
      p.start()
      q.put(2)  # This should block until consumer gets item 1.
      self.assertEqual(q.qsize(), 1)
      self.assertEqual(q.get(), 2)
      p.join()
      self.assertEqual(list(result_list), [1])

  def test_put_blocks_until_max_size_increases_from_process(self):
    ctx = multiprocessing.get_context("fork")
    q = variable_size_queue.VariableSizeMultiprocessingQueue(1, ctx=ctx)
    q.put(1)

    def increase_max_size(q):
      time.sleep(0.1)
      q.set_max_size(2)

    p = ctx.Process(target=increase_max_size, args=(q,))
    p.start()
    # This should block until max_size is increased in the other process.
    q.put(2)
    self.assertEqual(q.qsize(), 2)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)
    p.join()

  def test_empty(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    self.assertTrue(q.empty())
    q.put(1, block=False)
    while q.empty():
      time.sleep(0.1)
    self.assertFalse(q.empty())
    q.get()
    self.assertTrue(q.empty())

  def test_get_nowait(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    with self.assertRaises(queue.Empty):
      q.get_nowait()
    q.put(1)
    while q.empty():
      time.sleep(0.1)
    self.assertEqual(q.get_nowait(), 1)
    with self.assertRaises(queue.Empty):
      q.get_nowait()

  def test_close_and_cancel_join_thread(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=multiprocessing.get_context("fork")
    )
    q.close()
    q.cancel_join_thread()


if __name__ == "__main__":
  absltest.main()
