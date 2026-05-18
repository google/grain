# Copyright 2024 Google LLC
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
"""Primitives for working with Dataset APIs.

Classes in this module are shared by Dataset implementations as well as public
Dataset API.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import enum
import typing
from typing import Generic, Protocol, TypeVar

import numpy as np


T = TypeVar("T")


@typing.runtime_checkable
class ShapeDtypeStructProtocol(Protocol):
  """Protocol for a structure that has a shape and dtype."""

  @property
  def shape(self) -> tuple[int | None, ...]:
    ...

  @property
  def dtype(self) -> np.DTypeLike:
    ...


@dataclasses.dataclass(slots=True, frozen=True)
class ShapeDtypeStruct(ShapeDtypeStructProtocol):
  """A structure that has a shape and dtype."""

  shape: tuple[int | None, ...]
  dtype: np.typing.DTypeLike


@typing.runtime_checkable
class RandomAccessDataSource(Protocol[T]):
  """Interface for datasets where storage supports efficient random access.

  This `Protocol` defines the contract for any custom data source injected into
  the PyGrain pipeline. Implementations do not need to inherit from this class
  directly; they only need to implement the required structural methods
  (`__len__` and `__getitem__`).

  Notes:
  **Checkpointing**: If used with `DataLoader`, `__repr__` has to be
  additionally implemented to support checkpointing.

  **Multiprocessing**: If used with multiprocessing, the instance must be fully
  picklable.

  Example:
    Implementing a minimal, checkpoint-safe custom data source::

      from grain.sources import RandomAccessDataSource

      class MyInMemorySource:
        def __init__(self, data: list):
          self._data = data
        def __len__(self) -> int:
          return len(self._data)
        def __getitem__(self, index: int):
          return self._data[index]
        def __repr__(self) -> str:
          # Required for PyGrain checkpointing with DataLoader
          return f"MyInMemorySource(size={len(self)})"

      source = MyInMemorySource(["a", "b", "c"])
      # source satisfies the RandomAccessDataSource protocol.
      assert isinstance(source, RandomAccessDataSource)
  """

  def __len__(self) -> int:
    """Returns the total number of records in the data source.

    Returns:
      int: The total count of accessible records.
    """

  def __getitem__(self, index: int) -> T:
    """Returns the value for the given index.

    This method must be thread-safe and deterministic.

    Note that a number of sources take `SupportsIndex` instead of `int` for
    `index`. Such sources will still support `int` index and pass the
    `isinstance` check with this protocol, but all new source implementations
    should use `int` directly.

    Arguments:
      index: An integer in `[0, len(self)-1]`.

    Returns:
      The corresponding record. File data sources often return the raw bytes but
      records can be any Python object.
    """


class SupportsBatchedReadRandomAccessDataSource(
    RandomAccessDataSource[T], Protocol[T]
):
  """Interface for sources that support efficient random access and batched reads."""

  def _getitems(self, indices: Sequence[int]) -> Sequence[T]:
    """Returns the values for the given record_keys.

    This method must be threadsafe and deterministic.

    Arguments:
      indices: A sequence of integers in [0, len(self)-1].

    Returns:
      The sequence of corresponding records.
    """


class DatasetSelectionMap(abc.ABC):
  """Map from index to (constituent dataset index, index within dataset).

  This abstract base class defines the interface for mapping a single global
  sequence index across multiple underlying (constituent) datasets. It acts
  as a routing table for mixed or concatenated data pipelines.

  Note, this must be stateless, picklable and should avoid randomness to
  support determinism since it may be created and called concurrently in
  multiple processes.

  Example:
    Implementing a simple concatenation map for two datasets of size 2 and 3::

      import grain

      class ConcatMap(grain.transforms.DatasetSelectionMap):
        def __len__(self) -> int:
          return 7
        def __getitem__(self, index: int) -> tuple[int, int]:
          if index >= len(self):
            raise IndexError("Index out of range")
          if index < 3:
            return (0, index)
          else:
            return (1, index - 3)

      cmap = ConcatMap()
      assert len(cmap) == 7
      assert cmap[3] == (1, 0)
  """

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the length of this dataset.

    Returns:
      int: The total number of addressable records across all constituent
      datasets managed by this map.
    """

  @abc.abstractmethod
  def __getitem__(self, index: int) -> tuple[int, int]:
    """Returns constituent dataset index and index within this dataset.

    This method acts as a deterministic router, taking a global index and
    resolving it to a specific dataset and a local offset.

    Args:
      index: The global integer index being queried.

    Returns:
      tuple[int, int]: A two-element tuple containing:
        - The integer index of the target constituent dataset.
        - The local integer index within that specific dataset.
    """


@enum.unique
class ExecutionTrackingMode(enum.Enum):
  """Represents different modes for tracking execution statistics.

  Available modes:
    DISABLED:
      No execution statistics are measured. This mode is the default.
    STAGE_TIMING:
      The time taken for each transformation stage to execute is measured and
      recorded. This recorded time reflects the duration spent within the
      specific transformation to return an element, excluding the time spent in
      any parent transformations. The recorded time can be retrieved using
      `grain.experimental.get_execution_summary` method.

  Example:
    To enable stage timing, set `execution_tracking_mode` in
    `grain.experimental.DatasetOptions` and pass it to your dataset pipeline
    using `grain.experimental.WithOptionsIterDataset` or
    `grain.experimental.WithOptionsMapDataset`::

      import grain

      options = grain.experimental.DatasetOptions(
          execution_tracking_mode=grain.experimental.ExecutionTrackingMode.STAGE_TIMING
      )
      ds = grain.MapDataset.range(10).to_iter_dataset()
      ds_with_stage_timing = (
          grain.experimental.WithOptionsIterDataset(ds, options)
      )

      for element in ds_with_stage_timing:
        print(element)
  """

  DISABLED = enum.auto()
  STAGE_TIMING = enum.auto()


@typing.runtime_checkable
class SupportsSharedMemoryOutput(Protocol):
  """Protocol for datasets that support shared memory output.

  Currently, only Batch transform supports it. The primary use-case for this is
  to support directly passing NumPy arrays between processes without needing to
  perform an additional copy.
  """

  def enable_shared_memory_output(self) -> None:
    """Enables shared memory output for the dataset."""
    ...


@dataclasses.dataclass(slots=True, frozen=True)
class _Default(Generic[T]):
  """Default options value holder."""

  value: T


@dataclasses.dataclass(kw_only=True, frozen=True)
class DatasetOptions:
  # pyformat: disable
  """Holds options used by dataset transformations.

  This dataclass manages execution, telemetry, and performance parameters
  for PyGrain data pipelines. It tracks which fields are explicitly set by
  the user versus which fallback to default values, enabling intelligent
  option merging across different pipeline stages.

  Attributes:
    filter_warn_threshold_ratio: If the ratio of filtered out elements is above
      these thresholds, a warning will be issued. Value `None` disables the
      check. The ratio is calculated on non-overlapping windows of 1000
      elements. For instance, with `filter_warn_threshold_ratio=0.9` and 901
      elements out of the first 1000 (or elements 1000...2000) filtered out, a
      warning will be issued.
    filter_raise_threshold_ratio: If the ratio of filtered out elements is above
      these thresholds, an exception will be issued. Value `None` disables the
      check.
    execution_tracking_mode: The collection of execution statistics like total
      processing time taken by each transformation, number of elements produced
      etc. can be managed through various modes. If `DISABLED`, no statistics
      are collected.If `STAGE_TIMING`, the time it takes to process each
      transormation is collected. See `ExecutionTrackingMode` for more details.
    min_shm_size: The minimum size below which numpy arrays will copied between
      processes rather than passed via shared memory. For smaller arrays, the
      overhead of using shared memory can be higher than the cost of copying.

  Example:
    Applying custom options to dataset transformations::

      import grain

      ds = (
          grain.MapDataset.range(0, 1000)
          .filter(lambda x: x % 2 == 0)
          .to_iter_dataset()
      )
      # apply the DatasetOptions to create another IterDataset.
      ds_options = grain.experimental.DatasetOptions(filter_raise_threshold_ratio=0.1)
      ds = grain.experimental.WithOptionsIterDataset(ds, ds_options)
  """
  # pyformat: enable

  filter_warn_threshold_ratio: float | None | _Default[float] = _Default(0.9)
  filter_raise_threshold_ratio: float | None | _Default[None] = _Default(None)
  execution_tracking_mode: (
      ExecutionTrackingMode | _Default[ExecutionTrackingMode]
  ) = _Default(ExecutionTrackingMode.DISABLED)
  min_shm_size: int | _Default[int] = _Default(0)
  # Internal fields.

  # Names of fields which were set by the user.
  _user_set_fields: set[str] = dataclasses.field(
      default_factory=set, init=False
  )

  def __post_init__(self):
    # Replace default value objects with actual values.
    for field in dataclasses.fields(DatasetOptions):
      value = getattr(self, field.name)
      if isinstance(value, _Default):
        super().__setattr__(field.name, value.value)
      elif field.init:
        self._user_set_fields.add(field.name)

  def merge(self, other: DatasetOptions | None) -> DatasetOptions:
    """Merges these options with another DatasetOptions instance.

    This merge logic respects explicit user configurations over defaults.
    Explicitly set options in `self` take highest precedence, followed by
    explicitly set options in `other`, followed by the class default values.

    Args:
      other: Another DatasetOptions instance to merge into this one. If `None`,
        this method returns the current instance unmodified.

    Returns:
      DatasetOptions: A new DatasetOptions instance containing the merged
        configuration values.

    Example:
      Demonstrating the explicit-set precedence during a merge::

        import grain
        # opt1 explicitly sets min_shm_size
        opt1 = grain.experimental.DatasetOptions(min_shm_size=1024)

        # opt2 explicitly sets min_shm_size AND filter thresholds
        opt2 = grain.experimental.DatasetOptions(
            min_shm_size=512,
            filter_warn_threshold_ratio=None
        )

        # Merge opt2 into opt1. opt1's explicit values take precedence.
        merged = opt1.merge(opt2)
        assert merged.min_shm_size == 1024
        assert merged.filter_warn_threshold_ratio is None
    """
    if other is None:
      return self

    merged = {}
    for field in dataclasses.fields(DatasetOptions):
      if field.name in self._user_set_fields:
        merged[field.name] = getattr(self, field.name)
      elif field.name in other._user_set_fields:  # pylint: disable=protected-access
        merged[field.name] = getattr(other, field.name)
    return DatasetOptions(**merged)


@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class MultiprocessingContext:
  """Context of the current process as a part of multiprocessing system."""

  process_index: int = 0
  process_count: int = 1


@dataclasses.dataclass(kw_only=True, slots=True)
class IteratorContext:
  """Context shared by all iterators in a dataset.

  The context is mutable and:
    - Should be updated only before or during iterator initialization.
    - Attributes should only be used after all iterators in the pipeline are
      initialized. In practice, this means during pipeline execution with lazy
      initialization mechanisms such as `functools.cached_property`.
  """

  # Dataset transformation options.
  dataset_options: DatasetOptions = DatasetOptions()
  # Multiprocessing context of the worker process running this iterator.
  mp_context: MultiprocessingContext = MultiprocessingContext()
  # Whether this iterator is part of a DataLoader pipeline.
  is_dataloader_pipeline: bool = False

  def merge(self, other: IteratorContext) -> None:
    """Merges this context with the other in place."""
    self.dataset_options = self.dataset_options.merge(other.dataset_options)
    if self.mp_context != other.mp_context:
      raise ValueError(
          "Cannot merge contexts from different worker processes:"
          f" {self.mp_context} vs {other.mp_context}."
      )
    self.is_dataloader_pipeline = (
        self.is_dataloader_pipeline or other.is_dataloader_pipeline
    )
