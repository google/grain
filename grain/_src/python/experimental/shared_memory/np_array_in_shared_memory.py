"""The file contains functions to manage shared_memory usage for numpy array."""

import logging
from multiprocessing import reduction
from multiprocessing import shared_memory
import pickle
from typing import Any
import weakref

from absl import flags
import numpy as np


FLAGS = flags.FLAGS


# Arrays with fewer bytes with be serialized without using shared memory.
# This number is tuned based on the fact that imagenet starts to see benefit
# from batch_size 64 (9,633,792 bytes).
_MIN_BYTES_FOR_SHARED_MEMORY = 2**23


def _cleanup_shm(shm: shared_memory.SharedMemory) -> None:
  logging.info("Cleaning up shared memory %s.", shm.name)
  shm.close()
  shm.unlink()


def _rebuild_ndarray(serialized: bytes, shm_name: str) -> np.ndarray:
  logging.info("Rebuilding numpy array from shared memory %s.", shm_name)
  shm = shared_memory.SharedMemory(name=shm_name)
  arr = pickle.loads(serialized, buffers=[shm.buf])
  weakref.finalize(shm, _cleanup_shm, shm)
  return arr


def _reduce_ndarray(arr: Any):
  """Reduces a NumPy using shared memory when possible."""
  # We cannot move generic objects or non-continuous arrays to shared memory.
  # Moving small array is not worth the overhead.
  # Fall-back to default method for these three cases.
  is_small = arr.nbytes < _MIN_BYTES_FOR_SHARED_MEMORY
  if arr.dtype.hasobject or not arr.flags.c_contiguous or is_small:
    return arr.__reduce__()

  shm = shared_memory.SharedMemory(size=arr.nbytes, create=True)
  logging.info("Shared memory %s is created.", shm.name)

  def buffer_callback(arr_buf: pickle.PickleBuffer):
    shm.buf[:] = arr_buf.raw()[:]

  pickled_arr = pickle.dumps(
      arr, buffer_callback=buffer_callback, protocol=pickle.HIGHEST_PROTOCOL
  )

  shm.close()

  return _rebuild_ndarray, (pickled_arr, shm.name)


def enable_numpy_shared_memory_pickler() -> None:
  logging.info("Enabling shared memory pickler for numpy arrays...")
  reduction.ForkingPickler.register(np.ndarray, _reduce_ndarray)


# pylint:disable=protected-access
# pytype:disable=attribute-error
def disable_numpy_shared_memory_pickler() -> None:
  logging.info("Disabling shared memory pickler for numpy arrays...")
  if np.ndarray in reduction.ForkingPickler._extra_reducers:
    reduction.ForkingPickler._extra_reducers.pop(np.ndarray)


def numpy_shared_memory_pickler_enabled() -> bool:
  return np.ndarray in reduction.ForkingPickler._extra_reducers
# pytype:enable=attribute-error
# pylint:enable=protected-access
