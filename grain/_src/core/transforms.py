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
"""Abstract base classes for common types of transformations.

The idea is to implement atomic chunks of transformations as subclasses of
the base classes below (examples: resize image, tokenize text, add padding).
- More complex transformations can be created by chaining multiple
  transformations.
- It's recommended that subclasses are dataclasses.
- Libraries applying transformations can use these base classes to correctly
  apply transformations as part of Beam pipelines, data ingestion pipelines etc.
"""
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
import dataclasses
import inspect
from typing import Any, Union

import numpy as np


class MapTransform(abc.ABC):
  """Abstract base class for all 1:1 transformations of elements.

  Implementations should be threadsafe since they are often executed in
  parallel.
  """

  @abc.abstractmethod
  def map(self, element):
    """Maps a single element."""


class RandomMapTransform(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements.

  Implementations should be threadsafe since they are often executed in
  parallel.
  """

  @abc.abstractmethod
  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""


class MapWithIndex(abc.ABC):
  """Abstract base class for 1:1 transformations of elements and their index.

  Implementations should be threadsafe since they are often executed in
  parallel.
  """

  @abc.abstractmethod
  def map_with_index(self, index: int, element):
    """Maps a single element with its index."""


class TfRandomMapTransform(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements."""

  @abc.abstractmethod
  def np_random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""


class Filter(abc.ABC):
  """Abstract base class for filter transformations for individual elements.

  The pipeline will drop any element for which the filter function returns
  False.

  Implementations should be threadsafe since they are often executed in
  parallel.
  """

  @abc.abstractmethod
  def filter(self, element) -> bool:
    """Filters a single element; returns True if the element should be kept."""


class FlatMapTransform(abc.ABC):
  """Abstract base class for splitting operations of individual elements.

  Implementations should be threadsafe since they are often executed in
  parallel.

  Attributes
    max_fan_out: Absolute maximum number of splits that an element can generate.
    If element splits into number of sub-elements exceeding `max_fan_out`, an
    error is raised. In the case of variable fan-out, for performance reasons,
    please be mindful of the distribution in fan-outs. If the minimum and
    maximum fan-out in this distribution differ by several orders of magnitude,
    with a correspondingly very large `max_fan_out`, performance will degrade.
    In this case please consider preprocessing your data to keep the
    `max_fan_out` reasonable.
  """

  max_fan_out: int

  @abc.abstractmethod
  def flat_map(self, element) -> Sequence[Any]:
    """splits a single element."""


@dataclasses.dataclass(frozen=True)
class Batch:
  batch_size: int
  drop_remainder: bool = False


Transformation = Union[
    Batch,
    MapTransform,
    RandomMapTransform,
    TfRandomMapTransform,
    Filter,
    FlatMapTransform,
    MapWithIndex,
]
Transformations = Sequence[Transformation]


def get_pretty_transform_name(
    transform: Transformation | Callable[..., Any],
) -> str:
  """Returns a name for a transformation or callable with source file and line.

  Example: 'get_pretty_transform_name @ .../_src/core/transforms.py:116'

  Args:
    transform: The transfomation or callable to get the name of.
  """
  # We can't use `Transformation` here since `Union` does not support
  # `isinstance` check in Python 3.9.
  if isinstance(
      transform,
      (
          Batch,
          MapTransform,
          RandomMapTransform,
          TfRandomMapTransform,
          Filter,
          FlatMapTransform,
      ),
  ):
    # Check if transform class defines `__str__` and `__repr__` and use them if
    # so. Otherwise use the name of the class.
    if (
        transform.__class__.__str__ is object.__str__
        and transform.__class__.__repr__ is object.__repr__
    ):
      return transform.__class__.__name__
    return str(transform)

  # Some functions may not have `__name__`, e.g. `functools.partial`.
  transform_name = getattr(transform, "__name__", repr(transform))
  try:
    src_file = inspect.getsourcefile(transform)
    if src_file is None:
      return transform_name
    # If path is too long, shorten it to the last 3 parts.
    src_file_parts = src_file.split("/")
    if len(src_file_parts) >= 3:
      src_file = f".../{'/'.join(src_file_parts[-3:])}"
    src_lineno = inspect.getsourcelines(transform)[1]
    return f"{transform_name} @ {src_file}:{src_lineno}"
  except (OSError, TypeError):
    # `inspect` may raise if called on built-in functions.
    return transform_name
