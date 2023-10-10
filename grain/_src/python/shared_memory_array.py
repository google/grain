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
"""Shared memory array."""
from __future__ import annotations

import dataclasses
import math
import mmap
from multiprocessing import pool
from multiprocessing import shared_memory
import threading
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True, slots=True)
class SharedMemoryArrayMetadata:
  name: str
  shape: Iterable[int]
  dtype: npt.DTypeLike


class SharedMemoryArray(np.ndarray):
  """A NumPy array subclass which is backed by shared memory.

  This should be used in combination with Python multiprocessing.
  Compared with the normal NumPy ndarray it avoids expensive serialization
  when sending the array to another Python process (on the same machine).
  It also doesn't require a copy on the receiving side.

  The last processes using the array must call unlink_on_del()! Otherwise
  the memory will not be freed.
  """

  _unlink_thread_pool: pool.ThreadPool | None = None
  _unlink_semaphore: threading.Semaphore | None = None

  def __new__(
      cls,
      shape: Iterable[int],
      dtype: npt.DTypeLike = float,
  ):
    # See https://numpy.org/doc/stable/user/basics.subclassing.html
    size = math.prod(shape) * np.dtype(dtype).itemsize
    shm = shared_memory.SharedMemory(create=True, size=size)
    return cls.from_shared_memory(shm, shape, dtype)

  def __array_finalize__(self, obj):
    # This follows the `numpy.memmap` implementation
    if hasattr(obj, "shm") and np.may_share_memory(self, obj):
      self.shm = obj.shm
    else:
      self.shm = None
    self._unlink_on_del = getattr(obj, "_unlink_on_del", False)

  def __array_wrap__(self, obj, context=None):  # pylint: disable=unused-argument
    # This follows the `numpy.memmap` implementation
    if self is obj or type(self) is not type(self):
      return obj
    if not obj.shape:
      return obj[()]
    return obj.view(np.ndarray)

  @classmethod
  def from_metadata(
      cls, metadata: SharedMemoryArrayMetadata
  ) -> SharedMemoryArray:
    shm = shared_memory.SharedMemory(metadata.name)
    return cls.from_shared_memory(shm, metadata.shape, metadata.dtype)

  @property
  def metadata(self) -> SharedMemoryArrayMetadata:
    shm = self.shm
    assert isinstance(shm, shared_memory.SharedMemory)
    return SharedMemoryArrayMetadata(
        name=shm.name, shape=self.shape, dtype=self.dtype
    )

  @classmethod
  def from_shared_memory(
      cls,
      shm: shared_memory.SharedMemory,
      shape: Any,
      dtype: npt.DTypeLike,
  ) -> SharedMemoryArray:
    obj = super().__new__(cls, shape, dtype, buffer=shm.buf)
    obj.shm = shm
    obj._unlink_on_del = False
    return obj

  def __reduce_ex__(self, protocol):
    # For out-of-band pickling we don't need a PickleBuffer because the
    # `SharedMemory` class automatically pickles itself using only its name
    return self.from_shared_memory, (self.shm, self.shape, self.dtype)

  @classmethod
  def enable_async_del(cls, num_threads: int = 1) -> None:
    if not SharedMemoryArray._unlink_thread_pool:
      SharedMemoryArray._unlink_thread_pool = pool.ThreadPool(num_threads)
      SharedMemoryArray._unlink_semaphore = threading.Semaphore(num_threads)

  def unlink_on_del(self) -> None:
    """Mark this object responsible for unlinking the shared memory."""
    self._unlink_on_del = True

  def __del__(self) -> None:
    # Ensure that this array is not a view before closing shared memory
    if not isinstance(self.base, mmap.mmap):
      return
    thread_pool = SharedMemoryArray._unlink_thread_pool
    semaphore = SharedMemoryArray._unlink_semaphore
    shm = self.shm
    assert isinstance(shm, shared_memory.SharedMemory)
    if thread_pool:
      assert semaphore is not None
      # We use a semaphore to make sure that we don't accumulate too many
      # requests to close/unlink shared memory, which could lead to OOM errors
      semaphore.acquire()
      thread_pool.apply_async(shm.close, callback=lambda _: semaphore.release())
    else:
      shm.close()
    if self._unlink_on_del:
      if thread_pool:
        thread_pool.apply_async(shm.unlink)
      else:
        shm.unlink()
