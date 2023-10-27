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

# See http://shortn/_1qvUhTBlZM
RNG_STATE_ARRAY_MAX_SIZE = 4 * 8


def _reduce_ndarray(arr: np.ndarray):
  """Reduces a NumPy using shared memory when possible."""
  # There are three cases we fall-back to default reduce method.
  # 1. generic objects, 2. non-continuous arrays, 3. RecordMetadata case:
  # each Record contains a RecordMetadata which contains a RNG, and the
  # RNG keeps its internal state in the form of a numpy array. Unpickling
  # of the RNG would fail if pickled object is not of type numpy array.
  if (
      arr.dtype.hasobject
      or not arr.flags.c_contiguous
      or arr.nbytes <= RNG_STATE_ARRAY_MAX_SIZE
  ):
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
