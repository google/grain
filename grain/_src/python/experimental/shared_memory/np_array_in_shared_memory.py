"""The file contains functions to manage shared_memory usage for numpy array."""

from multiprocessing import reduction
from multiprocessing import shared_memory
import pickle

from absl import flags
from absl import logging
import numpy as np


FLAGS = flags.FLAGS


# Arrays with fewer bytes with be serialized without using shared memory.
# This number is tuned based on the fact that imagenet starts to see benefit
# from batch_size 64 (9,633,792 bytes).
_MIN_BYTES_FOR_SHARED_MEMORY = 2**23


def _cleanup_shm(shm: shared_memory.SharedMemory) -> None:
  logging.log_first_n(
      logging.INFO,
      "Cleaning up shared memory %s.",
      1,
      shm.name,
  )
  shm.close()
  shm.unlink()


def _rebuild_ndarray(serialized: bytes, shm_name: str) -> np.ndarray:
  """Rebuilds numpy array from serialized bytes using shared memory.

  Args:
    serialized: the serialized bytes
    shm_name: the name of the shared memory

  Returns:
    the numpy array.
  """
  logging.log_first_n(
      logging.INFO,
      "Rebuilding numpy array from shared memory %s.",
      1,
      shm_name,
  )
  shm = shared_memory.SharedMemory(name=shm_name)
  arr = pickle.loads(serialized, buffers=[shm.buf])
  arr_copied = arr.copy()
  # Need to delete shm's pointer first to avoid BufferError during closing shm.
  del arr
  _cleanup_shm(shm)
  return arr_copied


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
    return arr.__reduce__()  # pytype:disable=attribute-error

  shm = shared_memory.SharedMemory(size=arr.nbytes, create=True)
  logging.log_first_n(
      logging.INFO,
      "Shared memory %s is created.",
      1,
      shm.name,
  )

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
