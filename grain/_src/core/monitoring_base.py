# Copyright 2026 Google LLC
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
"""Grain metrics base definitions and protocols."""

from __future__ import annotations

import enum
from typing import Any, Protocol


@enum.unique
class Units(enum.Enum):
  """Grain metric units."""

  SECONDS = enum.auto()
  MILLISECONDS = enum.auto()
  MICROSECONDS = enum.auto()
  NANOSECONDS = enum.auto()
  BITS = enum.auto()
  BYTES = enum.auto()


class Metadata:
  """Grain metric metadata."""

  def __init__(self, description='', **kwargs):
    self.description = description
    for key, value in kwargs.items():
      setattr(self, key, value)
    self._kwargs = kwargs


class Bucketer:
  """Grain metric bucketer."""

  def __init__(self, *args, bucketer_type=None, **kwargs):
    self.args = args
    self.kwargs = kwargs
    self.type = bucketer_type

  @staticmethod
  def PowersOf(base: float):
    return Bucketer(base, bucketer_type='PowersOf')


class CounterProtocol(Protocol):
  """Protocol for Counter metrics."""

  def Increment(self, *args: Any, **kwargs: Any) -> None:
    ...

  def IncrementBy(self, value: float | int, *args: Any, **kwargs: Any) -> None:
    ...

  def Get(self, *args: Any, **kwargs: Any) -> float | int:
    ...

  def ClearAll(self) -> None:
    ...


class MetricProtocol(Protocol):
  """Protocol for Gauge metrics."""

  def Set(self, value: float | int, *args: Any, **kwargs: Any) -> None:
    ...

  def Get(self, *args: Any, **kwargs: Any) -> float | int:
    ...

  def ClearAll(self) -> None:
    ...


class EventMetricProtocol(Protocol):
  """Protocol for Event/Histogram metrics."""

  def Record(self, value: float | int, *args: Any, **kwargs: Any) -> None:
    ...

  def Get(self, *args: Any, **kwargs: Any) -> float | int:
    ...

  def ClearAll(self) -> None:
    ...
