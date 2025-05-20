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
import dataclasses
import enum
import typing
from typing import Generic, Protocol, TypeVar


T = TypeVar("T")


@typing.runtime_checkable
class RandomAccessDataSource(Protocol[T]):
  """Interface for datasets where storage supports efficient random access."""

  def __len__(self) -> int:
    ...

  def __getitem__(self, index: int) -> T:
    ...


class DatasetSelectionMap(abc.ABC):
  """Map from index to (constituent dataset index, index within dataset).

  Note, this must be stateless, picklable and should avoid randomness to
  support determinism since it may be created and called concurrently in
  multiple processes.
  """

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the length of this dataset."""

  @abc.abstractmethod
  def __getitem__(self, index: int) -> tuple[int, int]:
    """Returns constituent dataset index and index within this dataset."""


@enum.unique
class ExecutionTrackingMode(enum.Flag):
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
  """

  DISABLED = enum.auto()
  STAGE_TIMING = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class _Default(Generic[T]):
  """Default options value holder."""

  value: T


@dataclasses.dataclass(kw_only=True, frozen=True)
class DatasetOptions:
  # pyformat: disable
  """Holds options used by dataset transformations.

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
    """Merges these options with the other.

    Explicitly set options in `self` take precedence over options in `other`.

    Args:
      other: Options to merge.

    Returns:
      Merged options.
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

  def merge(self, other: IteratorContext) -> None:
    """Merges this context with the other in place."""
    self.dataset_options = self.dataset_options.merge(other.dataset_options)
    if self.mp_context != other.mp_context:
      raise ValueError(
          "Cannot merge contexts from different worker processes:"
          f" {self.mp_context} vs {other.mp_context}."
      )
