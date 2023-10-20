"""Methods to automatically use shared memory when sending NumPy arrays.

Methods below can register a custom reducer for `np.ndarray` classes. After
registration the reducer will turn NumPy arrays into our `SharedMemoryArray`
objects. This can speed up communication between worker and main process when
the last operation doesn't explicitly outputs `SharedMemoryArray`.

The serialized data only contains the metadata and the receiving process must
use it to reconstruct the array and free up the memory:
```
element: shared_memory_array.SharedMemoryArrayMetadata = ...
element = shared_memory_array.SharedMemoryArray.from_metadata(element)
# Free underlying memory when element reference count hits 0.
element.unlink_on_del()
```
"""
from multiprocessing import reduction

from absl import logging
from grain._src.python import shared_memory_array
import numpy as np


def _reduce_ndarray(arr: np.ndarray):
  """Reduces a NumPy using shared memory when possible."""
  # We cannot move generic objects or non-continuous arrays to shared memory.
  # Fall-back to default method for these three cases.
  if arr.dtype.hasobject or not arr.flags.c_contiguous:
    return arr.__reduce__()  # pytype: disable=attribute-error
  shared_arr = shared_memory_array.SharedMemoryArray(arr.shape, arr.dtype)
  logging.log_first_n(
      logging.INFO,
      "Shared memory %s is created.",
      1,
      shared_arr.shm.name,
  )
  np.copyto(shared_arr, arr, casting="no")
  return shared_arr.metadata.__reduce__()  # pytype: disable=attribute-error


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
