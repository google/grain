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


# pylint: disable=invalid-name
def get_metric(*args, **kwargs) -> NoOpMetric:
  return NoOpMetric(*args, **kwargs)


def record_event(*args):
  del args


counter_metric = value_metric = gauge_metric = get_metric
