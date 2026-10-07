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
"""This module provides variable size queue implementations."""

from multiprocessing import context
from multiprocessing import queues
from multiprocessing import reduction
import queue
import threading
import time
from typing import cast


class VariableSizeMultiprocessingQueue(queues.Queue):
  """A multiprocessing queue whose max size can be dynamically changed."""

  def __init__(
      self,
      max_size: int,
      ctx: context.BaseContext,
  ):
    super().__init__(maxsize=max_size, ctx=ctx)
    self._max_size_val = ctx.Value("i", max_size, lock=False)
    self._sem = ctx.Semaphore(max_size)
    self._resize_lock = ctx.Lock()
    self._pending_shrink = ctx.Value("i", 0, lock=False)

  def __getstate__(self):
    # pytype: disable=attribute-error
    context.assert_spawning(self)
    return cast(tuple, queues.Queue.__getstate__(self)) + (  # pylint: disable=g-bare-generic
        self._resize_lock,
        self._max_size_val,
        self._pending_shrink,
    )
    # pytype: enable=attribute-error

  def __setstate__(self, state):
    # pytype: disable=attribute-error
    queues.Queue.__setstate__(self, state[:-3])
    self._resize_lock, self._max_size_val, self._pending_shrink = state[-3:]
    # pytype: enable=attribute-error

  def set_max_size(self, max_size: int):
    """Sets the maximum size of the queue.

    This method can be used to dynamically change the capacity of the queue.
    If `max_size` is greater than the current size, the queue capacity is
    increased immediately. If `max_size` is smaller, the queue will prevent
    new items from being added until the number of items in the queue drops
    below the new `max_size`.

    Args:
      max_size: The new maximum size for the queue.
    """
    with self._resize_lock:
      delta = max_size - self._max_size_val.value
      if delta > 0:
        for _ in range(delta):
          self._sem.release()
        self._max_size_val.value = max_size
      elif delta < 0:
        self._max_size_val.value = max_size
        self._pending_shrink.value -= delta
        # Try to shrink capacity eagerly, but don't block.
        for _ in range(-delta):
          if self._sem.acquire(block=False):
            self._pending_shrink.value -= 1
          else:
            break

  def get(self, block: bool = True, timeout: float | None = None):
    """Gets an item from the queue, similar to `queue.Queue.get`."""
    # pytype: disable=attribute-error
    if self._closed:
      raise ValueError(f"Queue {self!r} is closed")
    if block and timeout is None:
      with self._rlock:
        res = self._recv_bytes()
    else:
      if block:
        deadline = time.time() + timeout
      if not self._rlock.acquire(block, timeout):
        raise queue.Empty
      try:
        if block:
          timeout = deadline - time.time()
          if not self._poll(timeout):
            raise queue.Empty
        elif not self._poll():
          raise queue.Empty
        res = self._recv_bytes()
      finally:
        self._rlock.release()
    # pytype: enable=attribute-error
    with self._resize_lock:
      if self._pending_shrink.value > 0:
        self._pending_shrink.value -= 1
      else:
        self._sem.release()
    return reduction.ForkingPickler.loads(res)


class VariableSizeQueue(queue.Queue):
  """A queue whose max size can be dynamically changed."""

  def __init__(self, max_size: int):
    super().__init__(maxsize=0)
    self._max_size = max_size
    self._cond = threading.Condition()

  def set_max_size(self, max_size: int):
    with self._cond:
      self._max_size = max_size
      self._cond.notify_all()

  def put(self, item, block: bool = True, timeout: float | None = None):
    """Puts an item into the queue, similar to `queue.Queue.put`.

    This method behaves like `queue.Queue.put`, but respects the current
    `_max_size` of this variable-size queue. If the queue is full based on
    `_max_size`, this method can block or raise `queue.Full` depending on
    `block` and `timeout`.

    Args:
      item: The object to put into the queue.
      block: If True, block until a free slot is available.
      timeout: If `block` is True, wait for at most `timeout` seconds.

    Raises:
      queue.Full: If the queue is full and `block` is False or the `timeout`
        is reached.
    """
    if not block:
      with self._cond:
        if self.qsize() >= self._max_size:
          raise queue.Full
        super().put(item, block=False)
      return

    deadline = None
    if timeout is not None:
      deadline = time.time() + timeout

    with self._cond:
      while self.qsize() >= self._max_size:
        if deadline is None:
          self._cond.wait()
          continue
        remaining = deadline - time.time()
        if remaining <= 0:
          raise queue.Full
        if not self._cond.wait(remaining):
          if self.qsize() >= self._max_size:
            raise queue.Full
          else:
            break
      super().put(item, block=False)

  def get(self, block: bool = True, timeout: float | None = None):
    item = super().get(block=block, timeout=timeout)
    with self._cond:
      self._cond.notify()
    return item
