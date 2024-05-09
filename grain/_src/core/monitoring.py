"""Grain metrics."""


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


Counter = Metric = EventMetric = NoOpMetric
