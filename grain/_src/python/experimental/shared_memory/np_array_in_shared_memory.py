"""The file contains functions to manage shared_memory usage for numpy array."""

from multiprocessing import reduction

from absl import flags
from absl import logging
from grain._src.python.shared_memory_array import SharedMemoryArray
import numpy as np


FLAGS = flags.FLAGS


# Arrays with fewer bytes with be serialized without using shared memory.
# This number is tuned based on the fact that imagenet starts to see benefit
# from batch_size 64 (9,633,792 bytes).
_MIN_BYTES_FOR_SHARED_MEMORY = 2**23


def _reduce_ndarray(arr: np.ndarray):
  """Reduces a NumPy using shared memory when possible."""
  # We cannot move generic objects or non-continuous arrays to shared memory.
  # Moving small array is not worth the overhead.
  # Fall-back to default method for these three cases.
  is_small = arr.nbytes < _MIN_BYTES_FOR_SHARED_MEMORY
  if arr.dtype.hasobject or not arr.flags.c_contiguous or is_small:
    logging.log_first_n(
        logging.INFO,
        "Small array with %d bytes, not using shared memory pickler.",
        1,
        arr.nbytes,
    )
    return arr.__reduce__()  # pytype: disable=attribute-error
  shared_arr = SharedMemoryArray(arr.shape, arr.dtype)
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
