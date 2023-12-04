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
"""This module defines common functions for multiprocessing/threading."""

import dataclasses
import multiprocessing
from multiprocessing import pool
import queue
from typing import TypeVar, Union, Callable

T = TypeVar('T')

_QUEUE_WAIT_TIMEOUT_SECONDS = 0.5
_ASYNC_RESULT_WAIT_TIMEOUT_SECONDS = 0.5


@dataclasses.dataclass
class _SystemTerminated:
  """When system terminates, this is returned instead of actual elements."""


SYSTEM_TERMINATED = _SystemTerminated()


def add_element_to_queue(
    element: T,
    elements_queue: queue.Queue[T],
    should_stop: Callable[[], bool],
) -> bool:
  """Try adding element to queue as long as should_stop() is not True.

  Args:
    element: Element to add.
    elements_queue: Target queue.
    should_stop: Callable to check whether addition should proceed (possibly
      after a re-try).

  Returns:
    Bool indicating whether addition was successfull.
  """
  while not should_stop():
    try:
      elements_queue.put(element, timeout=_QUEUE_WAIT_TIMEOUT_SECONDS)
      return True
    except queue.Full:
      pass
  return False


def get_element_from_queue(
    elements_queue: queue.Queue[T],
    should_stop: Callable[[], bool],
) -> Union[T, _SystemTerminated]:
  """Try getting element from queue as long as should_stop() is not True."""
  while not should_stop():
    try:
      return elements_queue.get(timeout=_QUEUE_WAIT_TIMEOUT_SECONDS)
    except queue.Empty:
      pass
  return SYSTEM_TERMINATED


def get_async_result(
    async_result: pool.AsyncResult[T],
    should_stop: Callable[[], bool],
) -> Union[T, _SystemTerminated]:
  """Wait for async result as long as should_stop() is not True."""
  while not should_stop():
    try:
      return async_result.get(timeout=_ASYNC_RESULT_WAIT_TIMEOUT_SECONDS)
    except multiprocessing.TimeoutError:
      pass
  return SYSTEM_TERMINATED
