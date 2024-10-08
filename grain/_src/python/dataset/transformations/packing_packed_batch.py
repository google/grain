"""This module provides a helper class for multi-bin first-fit packing.

Example packing is a step in many input pipelines for sequence to sequence
models where multiple examples are packed together as a single example in order
to maximise data fed to a TPU per batch. Our approach is implemented in pure
Python (thus easy to extend/ modify) and supports N-dimensional input features.

Note on the packing algorithm: We perform online packing. We start by
constructing an empty batch of "num_packing_bins" rows. For each input example,
we try to find the first row in the batch where it can be added. If the new
example can't be added, we construct a new batch to which the element is added.
This is equivalent to first-fit bin backing
(https://en.wikipedia.org/wiki/First-fit_bin_packing).
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import dataclasses
import sys
from typing import Generic, TypeVar

import jax
import jaxtyping as jt
import numpy as np

import tree


_T = TypeVar("_T")
_IS_PY310 = sys.version_info >= (3, 10)


@dataclasses.dataclass(
    **({"frozen": True, "kw_only": True} if _IS_PY310 else {"frozen": True})
)
class _SuccessfulRowOrFailingComponents:
  # Holds the index of the row to put a new element into if it can fit,
  # or None if it can't fit into any row.
  row: int | None
  # If it can't fit into any row, we return the name of all the components
  # that couldn't fit. We are packing multiple values, of which any of them
  # could fail to fit within a given bin. The values have string names like
  # "inputs" or "targets". This field holds the value of those components
  # that failed to fit.
  failing_components: list[str] | None


def _extract_and_rekey_packed_batch(
    values, *, segment_ids, positions, meta_features: Sequence[str]
):
  """Merges values, segment_ids and positions into a single struct.

  Args:
    values: The packed values.
    segment_ids: The segment IDs.
    positions: The positions.
    meta_features: Paths of meta features. For meta features segment_ids and
      positions will not be merged.

  Returns:
    A data element with packing meta features merged into it.
  """
  # Make a shallow copy of the values in the packed batch. The packed batch
  # structure is internal to the packing op which can be stateful, we should
  # not make assumptions that once the batch is emitted its contents won't be
  # used by the op.
  data = copy.copy(values)

  assert isinstance(values, dict)
  for k in list(values):
    if k not in meta_features:
      if not isinstance(segment_ids[k], np.ndarray):
        raise ValueError(
            f"Failed to extract segment ids for '{k}', which has type"
            f" {type(segment_ids[k])} rather than np.ndarray. Perhaps it should"
            " be marked as a meta feature?"
        )
      data[f"{k}_segment_ids"] = segment_ids[k].astype(np.int32)
      data[f"{k}_positions"] = positions[k].astype(np.int32)
  return data


def zeros(*args, **kwargs):
  return np.zeros(*args, **kwargs)


class PackedBatch(Generic[_T]):
  """Class to represent a batch of packed examples."""

  def __init__(
      self,
      element_for_shapes: jt.PyTree[np.ndarray],
      num_packing_bins: int,
      length_struct: jt.PyTree[int],
      meta_features: Sequence[str] = (),
  ):
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._meta_features = meta_features

    # Define the main buffers we will pack the data into.
    def make_packed_buffer(length: int, x: np.ndarray | int):
      is_scalar = np.ndim(x) == 0
      if is_scalar:
        shape = ()
        dtype = np.int64 if isinstance(x, int) else np.asarray(x).dtype
      else:
        assert isinstance(x, np.ndarray)
        shape = x.shape[1:]
        dtype = x.dtype
      return zeros(
          shape=(num_packing_bins, length, *shape),  # (B, T, ...)
          dtype=dtype,
      )

    self._values = jax.tree.map(
        make_packed_buffer, length_struct, element_for_shapes
    )

    def make_packed_aux_info(length: int):
      return zeros(shape=(num_packing_bins, length), dtype=np.int32)

    self._segment_ids = jax.tree.map(make_packed_aux_info, length_struct)
    self._positions = jax.tree.map(make_packed_aux_info, length_struct)

    # Tracks the next empty position to insert an example for each row
    # in the batch, for each feature in features_to_pack.
    self._first_free_cell_per_row = jax.tree.map(
        lambda _: zeros(num_packing_bins, dtype=np.int64), length_struct
    )

    # Tracks the number of examples already packed into row of the batch. Used
    # to fill the segmentation values for each feature.
    self._num_examples_per_row = [0 for _ in range(num_packing_bins)]

  def get_packed_batch(self):
    rows_with_values = sum(x > 0 for x in self._num_examples_per_row)
    if rows_with_values < len(self._num_examples_per_row):
      # Partial batch, last rows don't have values.
      self._values = jax.tree.map(lambda x: x[:rows_with_values], self._values)
      self._segment_ids = jax.tree.map(
          lambda x: x[:rows_with_values], self._segment_ids
      )
      self._positions = jax.tree.map(
          lambda x: x[:rows_with_values], self._positions
      )
    return _extract_and_rekey_packed_batch(
        self._values,
        segment_ids=self._segment_ids,
        positions=self._positions,
        meta_features=self._meta_features,
    )

  def _can_add_at_row(
      self, element: jt.PyTree[np.ndarray]
  ) -> _SuccessfulRowOrFailingComponents:
    """Checks whether the element can be added in any of the rows.

    Args:
      element: The element we are trying to fit into a row in the batch.

    Returns:
      SuccessfulRowOrFailingComponents: If the element fits into a row,
        return the index of that row. If it doesn't fit in any of the rows,
        return the names of the components that caused it to fail to fit.
    """
    element_feature_lengths = jax.tree.map(
        lambda x: 1 if np.ndim(x) == 0 else len(x), element
    )

    # Check no feature exceeds max length
    length_exceeded = jax.tree.map(
        lambda feature_length, max_length: feature_length > max_length,
        element_feature_lengths,
        self._length_struct,
    )
    if any(tree.flatten(length_exceeded)):
      raise ValueError(
          "Inputs to PackAndBatchOperation must be truncated to max length."
      )

    # For each row, check whether the total length after adding the current
    # element would exceed max feature lengths.
    def _feature_will_fit(feature_length, first_free_cell, max_length):
      return feature_length + first_free_cell <= max_length

    is_row_free_struct = tree.flatten_with_path(
        jax.tree.map(
            _feature_will_fit,
            element_feature_lengths,
            self._first_free_cell_per_row,
            self._length_struct,
        )
    )

    # Pick first row (if exists) where element can be added.
    for i in range(self._num_packing_bins):  # For each row.
      if all(free[i] for _, free in is_row_free_struct):
        # All components are free at that row.
        return _SuccessfulRowOrFailingComponents(row=i, failing_components=None)

    # There is no guarantee we have a single failing component, since one
    # component could be the reason an element could not fit in one row
    # and a different component could be the reason it could not fit in
    # a different row. In the event we have multiple, we return all of them
    # in order of number of rows they failed in, with highest number of failing
    # rows first.
    # Disabling the singleton comparison pylint because numpy does not work
    # without it.
    sorted_failing_components = sorted(
        [
            (component, np.count_nonzero(value == False))  # pylint: disable=singleton-comparison
            for component, value in is_row_free_struct
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    failing_components = [e[0] for e in sorted_failing_components if e[1] > 0]
    return _SuccessfulRowOrFailingComponents(
        row=None, failing_components=failing_components
    )

  def add_element_to_batch(
      self, element: jt.PyTree[np.ndarray], row: int
  ) -> None:
    """Adds element to current batch at the specified row."""
    # Apply updates to each feature.
    for per_feature_data in zip(
        tree.flatten(element),
        tree.flatten(self._values),
        tree.flatten(self._segment_ids),
        tree.flatten(self._positions),
        tree.flatten(self._first_free_cell_per_row),
    ):
      value, batch_value, segment_ids, positions, first_free_cell_per_row = (
          per_feature_data
      )
      value_length = 1 if np.ndim(value) == 0 else len(value)
      # Update batch value, segmentations, and positions.
      start = first_free_cell_per_row[row]
      end = first_free_cell_per_row[row] + value_length
      batch_value[row][start:end] = value
      segment_ids[row][start:end] = self._num_examples_per_row[row] + 1
      positions[row][start:end] = np.arange(end - start)
      # Update first_free_cell_per_row.
      first_free_cell_per_row[row] += value_length

    self._num_examples_per_row[row] += 1

  def try_add_to_batch(self, element) -> list[str] | None:
    """Finds a row in the batch at which element can be added.

    Args:
      element: The element we are trying to fit into a row in the batch.

    Returns:
      None if the element was successfully added to the batch. If the element
      could not be added, returns a list of strings indicating the components
      that failed.
    """
    successful_row_or_failing_component = self._can_add_at_row(element)
    successful_row = successful_row_or_failing_component.row
    failing_components = successful_row_or_failing_component.failing_components
    if successful_row is None:
      if not failing_components:
        raise ValueError(
            "If no successful row was found for an element, a failing component"
            " must be returned."
        )
      return failing_components
    self.add_element_to_batch(element, successful_row)

    return None
