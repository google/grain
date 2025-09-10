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
"""This module provides helper classes for multi-bin packing."""

from __future__ import annotations

import abc
import copy
import dataclasses
from typing import Any, Generic, List, Sequence, TypeVar

import numpy as np
import tree

from grain._src.core import tree_lib

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SuccessfulRowOrFailingComponents:
  # Holds the index of the row to put a new element into if it can fit,
  # or None if it can't fit into any row.
  row: int | None
  # If it can't fit into any row, we return the name of all the components
  # that couldn't fit.
  failing_components: list[str] | None


def _extract_and_rekey_packed_batch(
    values, *, segment_ids, positions, meta_features: Sequence[str]
):
  """Merges values, segment_ids and positions into a single struct."""
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


class PackedBatch(abc.ABC, Generic[_T]):
  """Base class to represent a batch of packed examples."""

  def __init__(
      self,
      element_for_shapes: Any,
      num_packing_bins: int,
      length_struct: Any,
      meta_features: Sequence[str] = (),
  ):
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._meta_features = meta_features

    def make_packed_buffer(length: int, x: np.ndarray | int):
      is_scalar = np.ndim(x) == 0
      shape = () if is_scalar else x.shape[1:]
      dtype = (
          (np.int64 if isinstance(x, int) else np.asarray(x).dtype)
          if is_scalar
          else x.dtype
      )
      return np.zeros(shape=(num_packing_bins, length, *shape), dtype=dtype)

    self._values = tree_lib.map_structure(
        make_packed_buffer, length_struct, element_for_shapes
    )
    self._segment_ids = tree_lib.map_structure(
        lambda l: np.zeros((num_packing_bins, l), dtype=np.int32),
        length_struct,
    )
    self._positions = tree_lib.map_structure(
        lambda l: np.zeros((num_packing_bins, l), dtype=np.int32),
        length_struct,
    )
    self._first_free_cell_per_row = tree_lib.map_structure(
        lambda _: np.zeros(num_packing_bins, dtype=np.int64), length_struct
    )
    self._num_examples_per_row = np.zeros(num_packing_bins, dtype=np.int32)

    # Flatten internal buffers and pre-calculate paths for efficient access.
    self._flat_paths_and_max = tree_lib.flatten_with_path(self._length_struct)
    self._feature_paths = [p for (p, _) in self._flat_paths_and_max]
    self._capacities = np.array(
        [int(m) for (_, m) in self._flat_paths_and_max], dtype=np.int64
    )
    self._flat_values = tree_lib.flatten(self._values)
    self._flat_seg_ids = tree_lib.flatten(self._segment_ids)
    self._flat_positions = tree_lib.flatten(self._positions)
    self._flat_ffcpr = tree_lib.flatten(self._first_free_cell_per_row)

  def get_packed_batch(self):
    """Returns the current packed batch, slicing off any empty trailing rows."""
    rows_with_values = np.count_nonzero(self._num_examples_per_row > 0)
    if rows_with_values < self._num_packing_bins:
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
      values, segment_ids, positions = self._values, self._segment_ids, self._positions

    return _extract_and_rekey_packed_batch(
        values,
        segment_ids=segment_ids,
        positions=positions,
        meta_features=self._meta_features,
    )

  def _get_element_lengths_flat(self, element: Any) -> np.ndarray:
    """Computes a flat vector of feature lengths for the given element."""
    flat_elem = tree_lib.flatten(element)
    return np.fromiter(
        ((1 if np.ndim(x) == 0 else len(x)) for x in flat_elem),
        dtype=np.int64,
        count=len(flat_elem),
    )

  def add_element_to_batch(self, element: Any, row: int) -> None:
    """Adds an element to the specified row using pre-flattened buffers."""
    flat_elem = tree_lib.flatten(element)
    seg_id = self._num_examples_per_row[row] + 1
    for idx, value in enumerate(flat_elem):
      value_len = 1 if np.ndim(value) == 0 else len(value)
      start = int(self._flat_ffcpr[idx][row])
      end = start + value_len

      self._flat_values[idx][row, start:end] = value
      self._flat_seg_ids[idx][row, start:end] = seg_id
      self._flat_positions[idx][row, start:end] = np.arange(
          end - start, dtype=np.int32
      )
      self._flat_ffcpr[idx][row] = end
    self._num_examples_per_row[row] += 1

  @abc.abstractmethod
  def try_add_to_batch(self, element: Any) -> list[str] | None:
    """Tries to add an element to the batch using a specific strategy."""
    raise NotImplementedError


class FirstFitPackedBatch(PackedBatch[_T]):
  """Implements first-fit packing of sequences."""

  def __init__(
      self,
      element_for_shapes: Any,
      num_packing_bins: int,
      length_struct: Any,
      meta_features: Sequence[str] = (),
  ):
    super().__init__(
        element_for_shapes,
        num_packing_bins,
        length_struct,
        meta_features=meta_features,
    )

  def try_add_to_batch(self, element: Any) -> list[str] | None:
    tree_lib.assert_same_structure(element, self._length_struct)
    element_lengths = self._get_element_lengths_flat(element)

    # Check if any feature exceeds its max length before attempting to pack.
    too_long = element_lengths > self._capacities
    if np.any(too_long):
      idxs = np.nonzero(too_long)[0]
      details = [
          (self._feature_paths[i], int(element_lengths[i]), int(self._capacities[i]))
          for i in idxs
      ]
      raise ValueError(
          "Inputs to packer must be truncated to max length. "
          f"Exceeds: (feature_path, feature_length, max_length) = {details}"
      )

    F = len(self._flat_ffcpr)
    fits_FB = np.empty((F, self._num_packing_bins), dtype=bool)
    for f in range(F):
      fits_FB[f, :] = (element_lengths[f] + self._flat_ffcpr[f]) <= self._capacities[f]

    feasible_rows = np.all(fits_FB, axis=0)
    if np.any(feasible_rows):
      row = int(np.argmax(feasible_rows))
      self.add_element_to_batch(element, row)
      return None

    fail_counts = np.sum(~fits_FB, axis=1)
    order = np.argsort(-fail_counts)
    failing_components = [self._feature_paths[i] for i in order if fail_counts[i] > 0]
    if not failing_components:
        raise ValueError("A failing component must be returned if no row is found.")
    return failing_components


class BestFitPackedBatch(PackedBatch[_T]):
  """Implements best-fit packing of sequences."""

  def __init__(
      self,
      element_for_shapes: Any,
      num_packing_bins: int,
      length_struct: Any,
      meta_features: Sequence[str] = (),
  ):
    super().__init__(
        element_for_shapes,
        num_packing_bins,
        length_struct,
        meta_features=meta_features,
    )

  def try_add_to_batch(self, element: Any) -> list[str] | None:
    tree_lib.assert_same_structure(element, self._length_struct)
    element_lengths = self._get_element_lengths_flat(element)

    # Check if any feature exceeds its max length before attempting to pack.
    too_long = element_lengths > self._capacities
    if np.any(too_long):
      idxs = np.nonzero(too_long)[0]
      details = [
          (self._feature_paths[i], int(element_lengths[i]), int(self._capacities[i]))
          for i in idxs
      ]
      raise ValueError(
          "Inputs to packer must be truncated to max length. "
          f"Exceeds: (feature_path, feature_length, max_length) = {details}"
      )

    free_cells_matrix = np.stack(self._flat_ffcpr, axis=0)
    new_free_cells = free_cells_matrix + element_lengths[:, np.newaxis]
    fittable_mask = np.all(new_free_cells <= self._capacities[:, np.newaxis], axis=0)

    if not np.any(fittable_mask):
      fits_FB = new_free_cells <= self._capacities[:, np.newaxis]
      fail_counts = np.sum(~fits_FB, axis=1)
      order = np.argsort(-fail_counts)
      failing_components = [self._feature_paths[i] for i in order if fail_counts[i] > 0]
      if not failing_components:
          raise ValueError("A failing component must be returned if no row is found.")
      return failing_components

    # Score is the sum of free cells (higher score = tighter fit).
    # We invalidate the scores of non-fittable bins by setting them to -1.
    scores = np.sum(free_cells_matrix, axis=0)
    scores[~fittable_mask] = -1

    best_bin_index = int(np.argmax(scores))
    self.add_element_to_batch(element, best_bin_index)
    return None
