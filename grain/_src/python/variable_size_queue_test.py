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

import platform
import queue
import threading
import time

from absl.testing import absltest
import multiprocessing as mp
from grain._src.python import variable_size_queue


def _consumer_function_for_test(q, result):
  time.sleep(0.1)
  result.append(q.get())


def _increase_max_size_function_for_test(q):
  time.sleep(0.1)
  q.set_max_size(2)


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

  def test_set_max_size_to_decrease_capacity_blocks_put(self):
    q = variable_size_queue.VariableSizeQueue(max_size=2)
    q.put(1)
    q.put(2)
    q.set_max_size(1)

    put_event = threading.Event()

    def _blocking_put():
      q.put(3)
      put_event.set()

    t = threading.Thread(target=_blocking_put)
    t.start()

    # The queue size is 2, max_size is 1. The put(3) call should block.
    # We wait a bit to ensure the thread has started and blocked on put().
    time.sleep(0.2)
    self.assertFalse(put_event.is_set())

    # Get one item. qsize becomes 1, which equals max_size.
    # However, because of _pending_shrink, no capacity is released,
    # so put(3) should still be blocked.
    self.assertEqual(q.get(), 1)
    time.sleep(0.2)
    self.assertFalse(put_event.is_set())

    # Get another item. qsize becomes 0.
    # This time, capacity should be released, unblocking put(3).
    self.assertEqual(q.get(), 2)
    self.assertTrue(put_event.wait(timeout=1))
    self.assertEqual(q.get(), 3)
    t.join()


class VariableSizeMultiprocessingQueueTest(absltest.TestCase):

  def test_put_and_get(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=mp.get_context("spawn")
    )
    q.put(1)
    self.assertEqual(q.get(), 1)

  def test_put_non_blocking_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=mp.get_context("spawn")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)

  def test_put_blocking_with_timeout_to_full_queue_raises_full(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=mp.get_context("spawn")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=True, timeout=0.1)

  def test_set_max_size_to_increase_capacity(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=mp.get_context("spawn")
    )
    q.put(1)
    with self.assertRaises(queue.Full):
      q.put(2, block=False)
    q.set_max_size(2)
    q.put(2, block=False)  # Should not raise.
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)

  def test_set_max_size_to_decrease_capacity(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        2, ctx=mp.get_context("spawn")
    )
    q.put(1)
    q.put(2)
    q.set_max_size(1)
    # qsize is 2, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 1)
    # qsize is 1, max_size is 1. put should fail.
    with self.assertRaises(queue.Full):
      q.put(3, block=False)
    self.assertEqual(q.get(), 2)
    # qsize is 0, max_size is 1. put should succeed.
    q.put(3, block=False)
    self.assertEqual(q.get(), 3)

  def test_set_max_size_to_decrease_capacity_blocks_put(self):
    ctx = mp.get_context("spawn")
    q = variable_size_queue.VariableSizeMultiprocessingQueue(2, ctx=ctx)
    q.put(1)
    q.put(2)
    q.set_max_size(1)

    put_event = threading.Event()

    def _blocking_put():
      q.put(3)
      put_event.set()

    t = threading.Thread(target=_blocking_put)
    t.start()

    # The queue has 2 items, max_size is 1. The put(3) call should block.
    # We wait a bit to ensure the thread has started and blocked on put().
    time.sleep(0.2)
    self.assertFalse(put_event.is_set())

    # Get one item. conceptual size becomes 1, which equals max_size.
    # However, because of _pending_shrink, no capacity is released,
    # so put(3) should still be blocked.
    self.assertEqual(q.get(), 1)
    time.sleep(0.2)
    self.assertFalse(put_event.is_set())

    # Get another item. conceptual size becomes 0.
    # This time, capacity should be released, unblocking put(3).
    self.assertEqual(q.get(), 2)
    self.assertTrue(put_event.wait(timeout=1))
    self.assertEqual(q.get(), 3)
    t.join()

  @absltest.skipIf(platform.system() == "Darwin", "Fails on macos-14 runner.")
  def test_put_blocks_until_item_is_retrieved_from_process(self):
    ctx = mp.get_context("spawn")
    q = variable_size_queue.VariableSizeMultiprocessingQueue(1, ctx=ctx)
    q.put(1)

    with ctx.Manager() as manager:
      result_list = manager.list()

      p = ctx.Process(target=_consumer_function_for_test, args=(q, result_list))
      p.start()
      q.put(2)  # This should block until consumer gets item 1.
      self.assertEqual(q.get(), 2)
      p.join()
      self.assertEqual(list(result_list), [1])

  def test_put_blocks_until_max_size_increases_from_process(self):
    ctx = mp.get_context("spawn")
    q = variable_size_queue.VariableSizeMultiprocessingQueue(1, ctx=ctx)
    q.put(1)

    p = ctx.Process(target=_increase_max_size_function_for_test, args=(q,))
    p.start()
    # This should block until max_size is increased in the other process.
    q.put(2)
    self.assertEqual(q.get(), 1)
    self.assertEqual(q.get(), 2)
    p.join()

  def test_empty(self):
    q = variable_size_queue.VariableSizeMultiprocessingQueue(
        1, ctx=mp.get_context("spawn")
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
        1, ctx=mp.get_context("spawn")
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
        1, ctx=mp.get_context("spawn")
    )
    q.close()
    q.cancel_join_thread()


if __name__ == "__main__":
  absltest.main()
