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
"""Implements batch transformations."""

from __future__ import annotations

from collections.abc import Sequence
import concurrent.futures
import functools
import math
import pprint
import sys
from typing import Any, Callable, TypeVar, cast

from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain._src.python.dataset.transformations import filter as filter_ds
import numpy as np


T = TypeVar("T")
S = TypeVar("S")


def _is_batch_pushdown_experiment_enabled() -> bool:
  return False


def _is_parallel_batch_experiment_enabled():
  return False


# The threshold (in bytes) for falling back to the serial `np.stack`  batching
# implementation. If an `np.array` has a smaller size than this threshold, the
# serial `np.stack` implementation will be used instead even if the parallel
# batching experiment is enabled.
_PARALLEL_BATCHING_MIN_BYTES = 256 * 1024


class _MakeBatchParallel:
  """A callable class for batching sequences of structured data in parallel.

  This class provides a parallel implementation for batching. When an instance
  of this class is called with a sequence of elements (e.g., a list of
  NumPy arrays), it uses a `ThreadPoolExecutor` to parallelize the memory copy
  operations: for each leaf in the data structure, it pre-allocates a single
  large NumPy array to hold the batched results, and then submits tasks to the
  thread pool to copy each individual element into its corresponding slice of
  the final batch array.

  If the elements to be batched are not NumPy arrays, or if the thread pool is
  not available, it falls back to the standard serial `np.stack` operation.

  The class is designed to be a drop-in replacement for the standard batching
  `_make_batch` function and integrates with `tree_lib.map_structure` to handle
  arbitrarily nested data structures.

  Note: This is an experimental feature and is only active when the
   'EXP_parallel_batch' experiment is enabled in the Grain configuration.
  """

  def __init__(self):
    self._parallel_batch_executor = concurrent.futures.ThreadPoolExecutor()

  def __call__(self, values: Sequence[T]) -> T:
    def _batch_fn(*xs: Sequence[T]) -> T:
      # If the thread pool is not available or the elements are not NumPy
      # arrays, fall back to the standard serial `np.stack` operation.
      if (self._parallel_batch_executor is None) or not isinstance(
          xs[0], np.ndarray
      ):
        return np.stack(xs)
      xs = cast(Sequence[np.ndarray], xs)
      # Fall back to the standard serial `np.stack` operation if the length of
      # an individual np.array within `values` is smaller in size (measured in
      # bytes) than the threshold.
      if any(x.nbytes < _PARALLEL_BATCHING_MIN_BYTES for x in xs):
        return np.stack(xs)

      out = np.empty([len(xs), *xs[0].shape], dtype=xs[0].dtype)
      # For each input array, submit a parallel task to the thread pool to copy
      # the data into the corresponding slice of the output array.
      fs = []
      for i, x in enumerate(xs):
        fs.append(self._parallel_batch_executor.submit(out.__setitem__, i, x))
      # Wait for all submitted futures to complete (blocking operation).
      for f in fs:
        f.result()

      return out

    if not values:
      raise ValueError("Cannot batch 0 values. Please file a bug.")

    try:
      result = tree_lib.map_structure(_batch_fn, *values)
      return result
    except ValueError as e:
      # NumPy error message doesn't include actual shapes and dtypes. Provide a
      # more helpful error message.
      raise ValueError(
          "Expected all input elements to have the same structure but got:\n"
          f"{pprint.pformat(tree_lib.spec_like(values))}"
      ) from e

  def __reduce__(self):
    return (self.__class__, ())

  def __del__(self):
    if self._parallel_batch_executor:
      self._parallel_batch_executor.shutdown(wait=False, cancel_futures=True)
      self._parallel_batch_executor = None


def make_batch(values: Sequence[T]) -> T:
  """Returns a batch of values with a new batch dimension at the front."""
  if not values:
    raise ValueError("Cannot batch 0 values. Please file a bug.")

  try:
    return tree_lib.map_structure(lambda *xs: np.stack(xs), *values)

  except ValueError as e:
    # NumPy error message doesn't include actual shapes and dtypes. Provide a
    # more helpful error message.
    raise ValueError(
        "Expected all input elements to have the same structure but got:\n"
        f"{pprint.pformat(tree_lib.spec_like(values))}"
    ) from e


def batch_and_pad(
    values: Sequence[T], *, batch_size: int, pad_value: Any = 0
) -> T:
  """Batches the given values and, if needed, pads the batch to the given size.

  Can be passed to `ds.batch` as `batch_fn` to avoid the need to drop the
  remainder data and pad it instead.

  Example usage::

    ds = grain.MapDataset.range(1, 5)
    batch_size = 3
    batch_fn = functools.partial(
        grain.experimental.batch_and_pad, batch_size=batch_size)
    ds = ds.batch(batch_size, batch_fn=batch_fn)
    list(ds) == [np.ndarray([1, 2, 3]), np.ndarray([4, 0, 0])]

  Args:
    values: The values to batch.
    batch_size: Target batch size. If the number of values is smaller than this,
      the batch is padded with `pad_value` to the given size.
    pad_value: The value to use for padding.

  Returns:
    A batch of values with a new batch dimension at the front.
  """
  if not values:
    raise ValueError("Cannot batch 0 values.")
  elif len(values) > batch_size:
    raise ValueError(f"Cannot batch {len(values)} values to {batch_size}.")
  elif len(values) == batch_size:
    return make_batch(values)
  expanded_values = tree_lib.map_structure(
      lambda x: np.expand_dims(x, axis=0), values
  )
  padding_size = batch_size - len(values)
  padding = tree_lib.map_structure(
      lambda x: np.full(
          shape=(padding_size,) + x.shape[1:],
          fill_value=pad_value,
          dtype=x.dtype,
      ),
      expanded_values[0],
  )
  try:
    return tree_lib.map_structure(
        lambda *xs: np.concatenate(xs, axis=0), *expanded_values, padding
    )
  except ValueError as e:
    # NumPy error message doesn't include actual shapes and dtypes. Provide a
    # more helpful error message.
    raise ValueError(
        "Expected all input elements to have the same structure but got:\n"
        f"{pprint.pformat(tree_lib.spec_like(values))}"
    ) from e


class _BatchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that batches elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[S],
      batch_size: int,
      drop_remainder: bool,
      batch_fn: Callable[[Sequence[S]], T],
  ):
    super().__init__(parent)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = batch_fn

  @stats.record_next_duration_if_output
  def __next__(self) -> T:
    values = []
    for _ in range(self._batch_size):
      try:
        values.append(next(self._parent))
      except StopIteration as e:
        if self._drop_remainder:
          raise e
        break
    if not values:
      raise StopIteration
    with self._stats.record_self_time():
      return self._stats.record_output_spec(self._batch_fn(values))

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return (
        f"BatchDatasetIterator(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


class BatchMapDataset(dataset.MapDataset[T]):
  """Batch transformation for non-sparse MapDatasets."""

  def __init__(
      self,
      parent: dataset.MapDataset[S],
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[S]], T] | None = None,
  ):
    """A MapDataset that batches elements.

    Args:
      parent: The parent MapDataset whose elements are batched.
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new batch dimension. If
        defined, the parallelized batch experiment will be disabled.
    """
    super().__init__(parent)
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    to_check = [parent]
    while to_check:
      next_ds = to_check.pop()
      if isinstance(next_ds, filter_ds.FilterMapDataset):
        raise ValueError(
            "`MapDataset.batch` can not follow `MapDataset.filter` "
            "because `filter` can discard elements. Convert `MapDataset` to "
            "`IterDataset` with `to_iter_dataset()` before calling `batch`."
        )
      to_check.extend(next_ds.parents)
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = make_batch if batch_fn is None else batch_fn
    if _is_parallel_batch_experiment_enabled() and batch_fn is None:
      self._batch_fn = _MakeBatchParallel()
    if self._drop_remainder:
      self._length = len(self._parent) // self._batch_size
    else:
      self._length = math.ceil(len(self._parent) / self._batch_size)

  @functools.cached_property
  def _get_parent_items_fn(self):
    # Leverage batch pushdown API to retrieve multiple items at once if the
    # experiment is enabled.
    if _is_batch_pushdown_experiment_enabled():
      return lambda items: self._parent._getitems(list(items))  # pylint: disable=protected-access
    return lambda items: [self._parent[i] for i in items]

  def __len__(self):
    return self._length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    # Each epoch gets batched separately. If users want to batch across epochs
    # they can repeat() before the batch().
    epoch, index_in_epoch = divmod(index, self._length)
    # Get range within the epoch without going outside the epoch.
    start = index_in_epoch * self._batch_size
    stop = min(len(self._parent), (index_in_epoch + 1) * self._batch_size)
    # Add offset for epoch.
    start += epoch * len(self._parent)
    stop += epoch * len(self._parent)
    values = self._get_parent_items_fn(range(start, stop))
    with self._stats.record_self_time():
      try:
        return self._stats.record_output_spec(self._batch_fn(values))
      except ValueError as e:
        if sys.version_info >= (3, 11):
          e.add_note(
              "\nIf you are trying to batch elements after a sparse "
              "transformation, such as `filter`, you need to first convert the "
              "dataset to `IterDataset` with `to_iter_dataset()` and then "
              "apply `batch`."
          )
        raise e

  def __str__(self) -> str:
    return (
        f"BatchMapDataset(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )


class BatchIterDataset(dataset.IterDataset[T]):
  """Batch transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset[S],
      batch_size: int,
      drop_remainder: bool = False,
      batch_fn: Callable[[Sequence[S]], T] | None = None,
  ):
    """A IterDataset that batches elements.

    Args:
      parent: The parent IterDataset whose elements are batched.
      batch_size: The number of elements to batch together.
      drop_remainder: Whether to drop the last batch if it is smaller than
        batch_size.
      batch_fn: A function that takes a list of elements and returns a batch.
        Defaults to stacking the elements along a new batch dimension. If
        defined, the parallelized batch experiment will be disabled.
    """
    super().__init__(parent)
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    self._batch_size = batch_size
    self._drop_remainder = drop_remainder
    self._batch_fn = make_batch if batch_fn is None else batch_fn
    if _is_parallel_batch_experiment_enabled() and batch_fn is None:
      self._batch_fn = _MakeBatchParallel()

  def __iter__(self) -> _BatchDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _BatchDatasetIterator(
        parent_iter,
        self._batch_size,
        drop_remainder=self._drop_remainder,
        batch_fn=self._batch_fn,
    )

  def __str__(self) -> str:
    return (
        f"BatchIterDataset(batch_size={self._batch_size},"
        f" drop_remainder={self._drop_remainder})"
    )
