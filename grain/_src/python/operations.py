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
"""This module defines operations that can be applied by the data transformer.

An operation takes as input an iterator and outputs an iterator. We provide
implementation for basic operations like MapOperation and FilterOperation (and
Batch to follow soon) but users can also implement their own operations.
"""
import dataclasses
from typing import Any, Callable, Generic, Iterator, Protocol, Sequence, TypeVar

from absl import logging
from grain._src.core import tree_lib
from grain._src.python import record
from grain._src.python.shared_memory_array import SharedMemoryArray
import numpy as np

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")


class Operation(Protocol):

  def __call__(
      self, input_iterator: Iterator[record.Record]
  ) -> Iterator[record.Record]:
    ...


@dataclasses.dataclass
class MapOperation(Generic[_IN, _OUT]):
  """Applies user-provided map_function to input records."""

  map_function: Callable[[_IN], _OUT]

  def __call__(
      self, input_iterator: Iterator[record.Record[_IN]]
  ) -> Iterator[record.Record[_OUT]]:
    logging.error(
        "Applying deprecated PyGrain MapOperation. Please use the"
        " grain.python.MapTransform."
    )
    for input_record in input_iterator:
      try:
        map_result = self.map_function(input_record.data)
      except Exception as e:
        raise ValueError(
            "PyGrain encountered an error when calling map function."
        ) from e
      yield record.Record(input_record.metadata.remove_record_key(), map_result)


@dataclasses.dataclass
class RandomMapOperation(Generic[_IN, _OUT]):
  """Applies user-provided random_map_function with rng to input records."""

  random_map_function: Callable[[_IN, np.random.Generator], _OUT]

  def __call__(
      self, input_iterator: Iterator[record.Record[_IN]]
  ) -> Iterator[record.Record[_OUT]]:
    logging.error(
        "Applying deprecated PyGrain RandomMapOperation. Please use the"
        " grain.python.RandomMapTransform."
    )
    for input_record in input_iterator:
      try:
        random_map_result = self.random_map_function(
            input_record.data, input_record.metadata.rng
        )
      except Exception as e:
        raise ValueError(
            "PyGrain encountered an error when calling random map function."
        ) from e
      yield record.Record(
          input_record.metadata.remove_record_key(), random_map_result
      )


@dataclasses.dataclass
class FilterOperation(Generic[_IN]):
  """Yields records from input iterator satisfying user-provided condition."""

  condition_function: Callable[[_IN], bool]

  def __call__(
      self, input_iterator: Iterator[record.Record[_IN]]
  ) -> Iterator[record.Record[_IN]]:
    logging.error(
        "Applying deprecated PyGrain FilterOperation. Please use the"
        " grain.python.FilterTransform."
    )
    for input_record in input_iterator:
      try:
        filter_result = self.condition_function(input_record.data)
      except Exception as e:
        raise ValueError(
            "PyGrain encountered an error when calling condition function."
        ) from e
      if filter_result:
        yield record.Record(
            input_record.metadata.remove_record_key(), input_record.data
        )


@dataclasses.dataclass
class BatchOperation(Generic[_IN, _OUT]):
  """Batches input examples into batches with given batch_size.

  Internally, examples are interpreted as JAX Pytrees. To batch records
  together, they must be of the same structure. Corresponding leaves are batched
  together into NumPy arrays. For more info about PyTrees, please refer to:
  https://jax.readthedocs.io/en/latest/pytrees.html.

  By default, we put Numpy arrays into Shared Memory. For more info about shared
  memory, please refer to:
  https://docs.python.org/3/library/multiprocessing.shared_memory.html
  """

  batch_size: int
  drop_remainder: bool = False

  def __post_init__(self):
    if self.batch_size <= 0:
      raise ValueError(
          f"batch_size must be a positive integer. Got {self.batch_size}."
      )
    self._use_shared_memory = False
    self._display_deprecation_message = True

  def __call__(
      self, input_iterator: Iterator[record.Record[_IN]]
  ) -> Iterator[record.Record[_OUT]]:
    if self._display_deprecation_message:  # pytype: disable=attribute-error
      logging.error(
          "Applying deprecated PyGrain BatchOperation. Please use the"
          " grain.python.Batch transformation."
      )
    records_to_batch = []
    last_record_metadata = None
    for input_record in input_iterator:
      last_record_metadata = input_record.metadata
      records_to_batch.append(input_record.data)
      if len(records_to_batch) == self.batch_size:
        batch = self._batch(records_to_batch)
        records_to_batch = []
        yield record.Record(last_record_metadata.remove_record_key(), batch)
    if records_to_batch and not self.drop_remainder:
      yield record.Record(
          last_record_metadata.remove_record_key(),  # pytype: disable=attribute-error
          self._batch(records_to_batch),
      )

  def _enable_shared_memory(self):
    self._use_shared_memory = True

  def disable_deprecation_message(self):
    self._display_deprecation_message = False

  def _validate_structure(self, input_records: Sequence[Any]) -> None:
    """Validate that all records have the same Pytree structure."""
    if not input_records:
      return
    first_record = input_records[0]
    non_matching_records_indices = []
    non_matching_records = []
    for index, input_record in enumerate(input_records[1:]):
      try:
        tree_lib.assert_same_structure(first_record, input_record)
      except ValueError:
        non_matching_records_indices.append(index + 1)
        non_matching_records.append(input_record)

    if non_matching_records:
      raise TypeError(
          "Record structures do not match. Record at position 0 has "
          f"structure {first_record}, while records at "
          f"positions {non_matching_records_indices} have "
          f"structures {non_matching_records}."
      )

  def _batch(self, input_records: Sequence[Any]):
    """Batches records together and copies Numpy arrays to Shared Memory."""
    self._validate_structure(input_records)

    def stacking_function(*args):
      first_arg = np.asanyarray(args[0])
      shape, dtype = (len(args),) + first_arg.shape, first_arg.dtype
      if not self._use_shared_memory or dtype.hasobject:
        return np.stack(args)
      return np.stack(args, out=SharedMemoryArray(shape, dtype=dtype)).metadata

    return tree_lib.map_structure(
        stacking_function, input_records[0], *input_records[1:]
    )
