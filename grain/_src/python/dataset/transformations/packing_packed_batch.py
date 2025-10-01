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
"""This module provides helper classes for multi-bin packing.

Example packing is a step in many input pipelines for sequence to sequence
models where multiple examples are packed together as a single example in order
to maximise data fed to a TPU per batch. Our approach is implemented in pure
Python (thus easy to extend/ modify) and supports N-dimensional input features.
Note on the packing algorithms: We perform online packing. We start by
constructing an empty batch of "num_packing_bins" rows. For each input example,
we try to find a bin where it can be added. If the new example can't fit in any
bin, the current batch is finalized, and a new batch is started with that
element. This module implements two common strategies:

- First-Fit: For a new example, this strategy finds the first available
  bin that it can fit into. This is implemented in FirstFitPackedBatch.
  (https://en.wikipedia.org/wiki/First-fit_bin_packing).
- Best-Fit: For a new example, this strategy checks all available bins and
  places it into the one that leaves the least amount of empty space (i.e., the
  tightest fit). This is implemented in BestFitPackedBatch.
  (https://en.wikipedia.org/wiki/Best-fit_bin_packing).
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import functools
from typing import Any, Generic, Sequence, TypeVar

from grain._src.core import tree_lib
import numpy as np


_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SuccessfulRowOrFailingComponents:
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


class PackedBatch(abc.ABC, Generic[_T]):
  """Base class to represent a batch of packed examples."""

  def __init__(
      self,
      element_for_shapes: Any,  # PyTree[np.ndarray]
      num_packing_bins: int,
      length_struct: Any,  # PyTree[int]
      meta_features: Sequence[str] = (),
      pack_alignment_struct: Any = None,
      padding_struct: Any = None,
      max_sequences_per_bin: int | None = None,
  ):
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._meta_features = meta_features
    self._size_bytes = 0
    self._max_sequences_per_bin = max_sequences_per_bin

    # Define the main buffers we will pack the data into.
    def make_packed_buffer(length: int, x: np.ndarray | int, padding: Any):
      is_scalar = np.ndim(x) == 0
      if is_scalar:
        shape = ()
        dtype = np.int64 if isinstance(x, int) else np.asarray(x).dtype
      else:
        assert isinstance(x, np.ndarray), type(x)
        shape = x.shape[1:]
        dtype = x.dtype
      buffer_fn = (
          zeros
          if padding is None
          else functools.partial(np.full, fill_value=padding)
      )
      buffer = buffer_fn(
          shape=(num_packing_bins, length, *shape),  # (B, T, ...)
          dtype=dtype,
      )
      self._size_bytes += buffer.nbytes
      return buffer

    if padding_struct is None:
      padding_struct = tree_lib.map_structure(lambda x: None, length_struct)

    self._values = tree_lib.map_structure(
        make_packed_buffer, length_struct, element_for_shapes, padding_struct
    )

    def make_packed_aux_info(length: int):
      buffer = zeros(shape=(num_packing_bins, length), dtype=np.int32)
      self._size_bytes += buffer.nbytes
      return buffer

    self._segment_ids = tree_lib.map_structure(
        make_packed_aux_info, length_struct
    )
    self._positions = tree_lib.map_structure(
        make_packed_aux_info, length_struct
    )
    if pack_alignment_struct is None:
      self._pack_alignments = tree_lib.map_structure(lambda x: 1, length_struct)
    else:
      self._pack_alignments = tree_lib.map_structure(
          lambda x: x, pack_alignment_struct
      )

    def _make_first_free_cell_per_row_buffer(_):
      buffer = zeros(num_packing_bins, dtype=np.int64)
      self._size_bytes += buffer.nbytes
      return buffer

    # Tracks the next empty position to insert an example for each row
    # in the batch, for each feature in features_to_pack.
    self._first_free_cell_per_row = tree_lib.map_structure(
        _make_first_free_cell_per_row_buffer, length_struct
    )

    # Tracks the number of examples already packed into row of the batch. Used
    # to fill the segmentation values for each feature and to make sure that
    # the maximum batches per row is not exceeded
    self._num_examples_per_row = zeros(num_packing_bins, dtype=np.int32)

    # Flatten internal buffers and pre-calculate paths for efficient access.
    self._flat_paths_and_max = tree_lib.flatten_with_path(self._length_struct)
    self._feature_paths = [p for (p, _) in self._flat_paths_and_max]
    self._capacities = np.array(
        [int(m) for (_, m) in self._flat_paths_and_max], dtype=np.int64
    )
    self._flat_values = tree_lib.flatten(self._values)
    self._flat_segment_ids = tree_lib.flatten(self._segment_ids)
    self._flat_positions = tree_lib.flatten(self._positions)
    self._flat_first_free_cell_per_row = tree_lib.flatten(
        self._first_free_cell_per_row
    )

  def get_size_bytes(self) -> int:
    """Returns the size of the packed batch in bytes."""
    return self._size_bytes

  def get_packed_batch(self):
    """Returns the current packed batch, slicing off any empty trailing rows."""
    rows_with_values = np.count_nonzero(self._num_examples_per_row > 0)
    if rows_with_values < self._num_packing_bins:
      # Partial batch, last rows don't have values.
      values = tree_lib.map_structure(
          lambda x: x[:rows_with_values], self._values
      )
      segment_ids = tree_lib.map_structure(
          lambda x: x[:rows_with_values], self._segment_ids
      )
      positions = tree_lib.map_structure(
          lambda x: x[:rows_with_values], self._positions
      )
    else:
      values, segment_ids, positions = (
          self._values,
          self._segment_ids,
          self._positions,
      )

    return _extract_and_rekey_packed_batch(
        values,
        segment_ids=segment_ids,
        positions=positions,
        meta_features=self._meta_features,
    )

  def _get_element_lengths_flat(self, element: Any) -> np.ndarray:
    """Computes a flat vector of feature lengths for the given element."""
    flat_element = tree_lib.flatten(element)
    return np.fromiter(
        ((1 if np.ndim(x) == 0 else len(x)) for x in flat_element),
        dtype=np.int64,
        count=len(flat_element),
    )

  def add_element_to_batch(self, element: Any, row: int) -> None:
    """Adds an element to the specified row using pre-flattened buffers."""
    flat_element = tree_lib.flatten(element)
    flat_alignments = tree_lib.flatten(self._pack_alignments)
    segment_id = self._num_examples_per_row[row] + 1

    for idx, value in enumerate(flat_element):
      value_length = 1 if np.ndim(value) == 0 else len(value)
      start = int(self._flat_first_free_cell_per_row[idx][row])
      end = start + value_length

      alignment = flat_alignments[idx]
      padded_end = ((end + alignment - 1) // alignment) * alignment

      self._flat_values[idx][row, start:end] = value
      self._flat_segment_ids[idx][row, start:end] = segment_id
      self._flat_positions[idx][row, start:end] = np.arange(
          end - start, dtype=np.int32
      )
      self._flat_first_free_cell_per_row[idx][row] = padded_end
    self._num_examples_per_row[row] += 1

  @abc.abstractmethod
  def try_add_to_batch(self, element: Any) -> list[str] | None:
    """Tries to add an element to the batch using a specific strategy."""
    raise NotImplementedError


class FirstFitPackedBatch(PackedBatch[_T]):
  """Implements first-fit packing of sequences."""

  def try_add_to_batch(self, element: Any) -> list[str] | None:
    tree_lib.assert_same_structure(element, self._length_struct)
    element_lengths = self._get_element_lengths_flat(element)

    # Check if any feature exceeds its max length before attempting to pack.
    too_long = element_lengths > self._capacities
    if np.any(too_long):
      idxs = np.nonzero(too_long)[0]
      details = [
          (
              self._feature_paths[i],
              int(element_lengths[i]),
              int(self._capacities[i]),
          )
          for i in idxs
      ]
      raise ValueError(
          "Inputs to PackedBatch must be truncated to max length. "
          f"Exceeds: (feature_path, feature_length, max_length) = {details}"
      )

    # For each feature and row, check if the element fits by length.
    num_features = len(self._flat_first_free_cell_per_row)
    features_fit_by_length = np.empty(
        (num_features, self._num_packing_bins), dtype=bool
    )
    for i in range(num_features):
      features_fit_by_length[i, :] = (
          element_lengths[i] + self._flat_first_free_cell_per_row[i]
      ) <= self._capacities[i]

    # Combine with max sequences per bin constraint to get overall fit matrix.
    if self._max_sequences_per_bin is not None:
      # A bin is not full if it has less than `max_sequences_per_bin`` examples.
      not_full_mask = self._num_examples_per_row < self._max_sequences_per_bin
      # A feature "fits" in a bin only if it fits by length AND the bin is not
      # full.
      features_fit = np.logical_and(features_fit_by_length, not_full_mask)
    else:
      features_fit = features_fit_by_length

    # Find the first row where all features fit.
    feasible_rows = np.all(features_fit, axis=0)
    if np.any(feasible_rows):
      row = int(np.argmax(feasible_rows))
      self.add_element_to_batch(element, row)
      return None

    # There is no guarantee we have a single failing component, since one
    # component could be the reason an element could not fit in one row
    # and a different component could be the reason it could not fit in
    # a different row. In the event we have multiple, we return all of them
    # in order of number of rows they failed in, with highest number of failing
    # rows first.
    fail_counts = np.sum(~features_fit, axis=1)
    order = np.argsort(-fail_counts)
    failing_components = [
        self._feature_paths[i] for i in order if fail_counts[i] > 0
    ]
    if not failing_components:
      raise ValueError(
          "A failing component must be returned if no row is found."
      )
    return failing_components


class BestFitPackedBatch(PackedBatch[_T]):
  """Implements best-fit packing of sequences."""

  def try_add_to_batch(self, element: Any) -> list[str] | None:
    tree_lib.assert_same_structure(element, self._length_struct)
    element_lengths = self._get_element_lengths_flat(element)

    # Check if any feature exceeds its max length before attempting to pack.
    too_long = element_lengths > self._capacities
    if np.any(too_long):
      indices = np.nonzero(too_long)[0]
      details = [
          (
              self._feature_paths[i],
              int(element_lengths[i]),
              int(self._capacities[i]),
          )
          for i in indices
      ]
      raise ValueError(
          "Inputs to packer must be truncated to max length. "
          f"Exceeds: (feature_path, feature_length, max_length) = {details}"
      )

    free_cells_matrix = np.stack(self._flat_first_free_cell_per_row, axis=0)
    new_free_cells = free_cells_matrix + element_lengths[:, np.newaxis]

    if self._max_sequences_per_bin is not None:
      # Invalidate bins that have reached the max sequence limit. Adding the
      # capacity ensures they fail the subsequent length check.
      max_sequence_mask = np.where(
          self._num_examples_per_row < self._max_sequences_per_bin, 0, 1
      )
      new_free_cells = (
          new_free_cells + max_sequence_mask * self._capacities[:, np.newaxis]
      )
    fittable_mask = np.all(
        new_free_cells <= self._capacities[:, np.newaxis], axis=0
    )

    if not np.any(fittable_mask):
      features_fit = new_free_cells <= self._capacities[:, np.newaxis]
      fail_counts = np.sum(~features_fit, axis=1)
      order = np.argsort(-fail_counts)
      failing_components = [
          self._feature_paths[i] for i in order if fail_counts[i] > 0
      ]
      if not failing_components:
        raise ValueError(
            "If no successful row was found for an element, a failing component"
            " must be returned."
        )
      return failing_components

    # Score is the sum of free cells (higher score = tighter fit).
    # We invalidate the scores of non-fittable bins by setting them to -1.
    scores = np.sum(free_cells_matrix, axis=0)
    scores[~fittable_mask] = -1

    best_bin_index = int(np.argmax(scores))
    self.add_element_to_batch(element, best_bin_index)
    return None
