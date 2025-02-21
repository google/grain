"""This module provides an implementation for example packing in pure python.

If using LazyDataset, please use APIs in
_src/python/dataset/transformations/packing.py.

Example packing is a step in many input pipelines for sequence to sequence
models where multiple examples are packed together as a single example in order
to maximise data fed to a TPU per batch. Our approach is implemented in pure
Python (thus easy to extend/ modify) and supports N-dimensional input features.

Note on the packing algorithm: We perform online packing. We start by
constructing an empty batch of "batch_size" rows. For each input example, we
try to find the first row in the batch where it can be added. If the new example
can't be added, we construct a new batch to which the element is added.
"""

import dataclasses
from typing import Any, Generic, Iterator, TypeVar, Union, cast

from absl import logging
from grain._src.core import tree_lib
from grain._src.python import record
import numpy as np

_T = TypeVar("_T")


class _PackedBatch:
  """Class to represent a batch of packed examples."""

  def __init__(
      self,
      element_for_shapes: Any,  # PyTree[np.ndarray]
      batch_size: int,
      length_struct: Any,  # PyTree[int]
  ):
    self._batch_size = batch_size
    self._length_struct = length_struct

    # Define the main buffers we will pack the data into.
    def make_packed_buffer(length: int, input_arr: np.ndarray):
      return np.zeros(
          shape=(batch_size, length, *input_arr.shape[1:]),  # (B, T, ...)
          dtype=input_arr.dtype,
      )

    self._batch = tree_lib.map_structure(
        make_packed_buffer, length_struct, element_for_shapes
    )

    def make_packed_aux_info(length: int):
      return np.zeros(shape=(batch_size, length), dtype=np.int32)

    self._segmentations = tree_lib.map_structure(
        make_packed_aux_info, length_struct
    )
    self._positions = tree_lib.map_structure(
        make_packed_aux_info, length_struct
    )

    # Tracks the next empty position to insert an example for each row
    # in the batch, for each feature in features_to_pack.
    self._first_free_cell_per_row = tree_lib.map_structure(
        lambda _: np.zeros(batch_size, dtype=np.int32), length_struct
    )

    # Tracks the number of examples already packed into row of the batch. Used
    # to fill the segmentation values for each feature.
    self._num_examples_per_row = [0 for _ in range(batch_size)]

    # For determinism, the metadata.index for the packed batch must match
    # metadata.index of the _last_ included input example.
    self._last_record_metadata = None

  def get_packed_batch(self) -> record.Record[tuple[_T, _T, _T]]:
    assert self._last_record_metadata is not None
    return record.Record(
        metadata=cast(record.RecordMetadata, self._last_record_metadata),
        data=(self._batch, self._segmentations, self._positions),
    )

  def _can_add_at_row(
      self,
      element: Any,  # PyTree[np.ndarray]
  ) -> int:
    """Returns the index of the first row which fits element, or -1 if none."""
    element_feature_lengths = tree_lib.map_structure(len, element)

    # Check no feature exceeds max length
    length_exceeded = tree_lib.map_structure(
        lambda feature_length, max_length: feature_length > max_length,
        element_feature_lengths,
        self._length_struct,
    )
    if any(tree_lib.flatten(length_exceeded)):
      raise ValueError(
          "Inputs to PackAndBatchOperation must be truncated to max length."
      )

    # For each row, check whether the total length after adding the current
    # element would exceed max feature lengths.
    def _feature_will_fit(feature_length, first_free_cell, max_length):
      return feature_length + first_free_cell <= max_length

    is_row_free_struct = tree_lib.map_structure(
        _feature_will_fit,
        element_feature_lengths,
        self._first_free_cell_per_row,
        self._length_struct,
    )

    ## Pick first row (if exists) where element can be added.
    for i in range(self._batch_size):
      row_is_free_per_feature = [
          free[i] for free in tree_lib.flatten(is_row_free_struct)
      ]
      if all(row_is_free_per_feature):
        return i
    return -1

  def add_element_to_batch(
      self,
      element: Any,  # PyTree[np.ndarray]
      row: int,
  ) -> None:
    """Adds element to current batch at the specified row."""
    # Apply updates to each feature.
    for per_feature_data in zip(
        tree_lib.flatten(element),
        tree_lib.flatten(self._batch),
        tree_lib.flatten(self._segmentations),
        tree_lib.flatten(self._positions),
        tree_lib.flatten(self._first_free_cell_per_row),
    ):
      value, batch_value, segmentations, positions, first_free_cell_per_row = (
          per_feature_data
      )
      # Update batch value, segmentations, and positions.
      start = first_free_cell_per_row[row]
      end = first_free_cell_per_row[row] + len(value)
      batch_value[row][start:end] = value
      segmentations[row][start:end] = self._num_examples_per_row[row] + 1
      positions[row][start:end] = np.arange(end - start)
      # Update first_free_cell_per_row.
      first_free_cell_per_row[row] += len(value)

    self._num_examples_per_row[row] += 1

  def try_add_to_batch(self, element: record.Record) -> bool:
    """Finds a row in the batch at which element can be added."""
    if (row_idx := self._can_add_at_row(element.data)) == -1:
      return False
    self.add_element_to_batch(element.data, row_idx)
    self._last_record_metadata = element.metadata.remove_record_key()
    return True


@dataclasses.dataclass
class PackAndBatchOperation(Generic[_T]):
  """PyGrain pack-and-batch operation - see module docstring.

  WARNING: This class is deprecated. Please use
  lazy_dataset.FirstFitPackIterDataset instead.

  Attributes:
    batch_size: int, the batch size.
    length_struct: A pytree, with the same structure as `input_iterator`
      elements, but where leaves are ints, representing the packed length of the
      corresponding feature.

  __call__() takes an input iterator, where elements are `Record`s containing:

    input_data: Pytrees of arrays. For more info about PyTrees, please refer to:
      https://jax.readthedocs.io/en/latest/pytrees.html. Packed leaves should be
      n-dimensional arrays, with sequence length as the leading dimension, i.e.
      shape (T_in, ...), where T_in < T_packed. Note that leaves can and will
      often have ragged length dimensions across different elements of the input
      iterator.

  The output of __call__() will be an iterator over `Record`s containing a
  3-tuple of Pytrees. These are:

    data: The batched and packed data. This is a Pytree with parallel structure
      to elements of `input_iterator`. Leaves have shape (B, T_packed, ...).
    segmentations: Pytree with the same structure as `data`, and leaves of shape
      (B, T). Represents which example each entry comes from. This may be used
      for Transformer attention masks, for example.
    positions: Pytree with the same structure as `data`, and leaves of shape
      (B, T). Represents the position of each entry within their original
      example. This may be used e.g. in Transformer absolute position
      embeddings.
  """

  length_struct: Any  # PyTree[int]
  batch_size: int
  # We don't know input shapes and corresponding buffer shapes until __call__.
  _cur_batch: Union[_PackedBatch, None] = None

  def __post_init__(self):
    logging.error(
        "PackAndBatchOperation is deprecated. Please use"
        " lazy_dataset.FirstFitPackIterDataset instead."
    )

  def __call__(
      self, input_iterator: Iterator[record.Record[_T]]
  ) -> Iterator[record.Record[tuple[_T, _T, _T]]]:
    for element in input_iterator:
      # Use `element` to set dtypes + trailing dimensions.
      if self._cur_batch is None:  # pytype: disable=attribute-error
        self._cur_batch = _PackedBatch(
            element.data, self.batch_size, self.length_struct
        )

      # Try adding element to the current packed batch.
      element_added_to_batch = self._cur_batch.try_add_to_batch(element)

      # When we have a full batch, yield the current packed data,
      # and then start a new batch with this element.
      if not element_added_to_batch:
        yield self._cur_batch.get_packed_batch()  # Main yield
        self._cur_batch = _PackedBatch(
            element.data, self.batch_size, self.length_struct
        )
        self._cur_batch.try_add_to_batch(element)

    # Final batch
    yield self._cur_batch.get_packed_batch()
