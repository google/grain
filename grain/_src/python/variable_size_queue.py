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
from multiprocessing import sharedctypes
import queue
import threading
import time
from typing import Any, cast


class VariableSizeMultiprocessingQueue(queues.Queue):
  """A multiprocessing queue whose max size can be dynamically changed."""

  def __init__(
      self,
      max_size: int | sharedctypes.Synchronized,
      ctx: context.BaseContext,
  ):
    super().__init__(maxsize=0, ctx=ctx)
    self._max_size = (
        max_size
        if isinstance(max_size, sharedctypes.Synchronized)
        else ctx.Value("i", max_size)
    )
    self._cond = ctx.Condition()

  def __getstate__(self):
    return cast(tuple[Any, ...], super().__getstate__()) + (
        self._max_size,
        self._cond,
    )

  def __setstate__(self, state):
    super().__setstate__(state[:-2])  # pytype: disable=attribute-error
    self._max_size, self._cond = state[-2:]

  def set_max_size(self, max_size: int):
    with self._cond:
      self._max_size.value = max_size
      self._cond.notify_all()

  def put(self, obj, block: bool = True, timeout: float | None = None):
    """Puts an item into the queue, similar to `queue.Queue.put`.

    This method behaves like `queue.Queue.put`, but respects the current
    `_max_size` of this variable-size queue. If the queue is full based on
    `_max_size`, this method can block or raise `queue.Full` depending on
    `block` and `timeout`.

    Args:
      obj: The object to put into the queue.
      block: If True, block until a free slot is available.
      timeout: If `block` is True, wait for at most `timeout` seconds.

    Raises:
      queue.Full: If the queue is full and `block` is False or the `timeout`
        is reached.
    """
    if not block:
      with self._cond:
        if self.qsize() >= self._max_size.value:
          raise queue.Full
        super().put(obj, block=False)
      return

    deadline = None
    if timeout is not None:
      deadline = time.time() + timeout

    with self._cond:
      while self.qsize() >= self._max_size.value:
        if deadline is None:
          self._cond.wait()
          continue
        remaining = deadline - time.time()
        if remaining <= 0:
          raise queue.Full
        if not self._cond.wait(remaining):
          if self.qsize() >= self._max_size.value:
            raise queue.Full
          else:
            break
      super().put(obj, block=False)

  def get(self, block: bool = True, timeout: float | None = None):
    item = super().get(block=block, timeout=timeout)
    with self._cond:
      self._cond.notify()
    return item

  def get_nowait(self):
    item = super().get_nowait()
    with self._cond:
      self._cond.notify()
    return item


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
