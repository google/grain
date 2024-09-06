"""Grain metrics."""

import enum


@enum.unique
class Units(enum.Enum):
  """Grain metric units."""

  NANOSECONDS = enum.auto()
  MILLISECONDS = enum.auto()


class NoOpMetric:
  """Grain metric no-op implementation."""

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def IncrementBy(self, *args, **kwargs):
    del args, kwargs

  def Increment(self, *args, **kwargs):
    self.IncrementBy(1, *args, **kwargs)

  def Set(self, *args, **kwargs):
    del args, kwargs

  def Record(self, *args, **kwargs):
    del args, kwargs

  def Get(self, *args, **kwargs):
    del args, kwargs


class Metadata:
  """Grain metric no-op metadata."""

  def __init__(self, *args, **kwargs):
    del args, kwargs


class Bucketer:
  """Grain metric no-op bucketer."""

  def __init__(self, *args, **kwargs):
    del args, kwargs

  def PowersOf(self, *args, **kwargs):
    del args, kwargs


Counter = Metric = EventMetric = NoOpMetric

def get_monitoring_root() -> None:
  return None
