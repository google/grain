"""Grain metrics."""


# pylint: disable=invalid-name
# pylint: disable=g-statement-before-imports
def __getattr__(name, *args, **kwargs):
  """Intercepts all attribute lookups and returns a no-op function."""
  del args, kwargs
  if name == 'Units':
    return Units
  return NoOp


class Units:
  """Grain metric units."""

  SECONDS = 'seconds'


class NoOp:
  """No-Op Grain metric."""

  def __init__(self, *args, **kwargs):
    """No-op initialization method."""
    pass

  def __getattribute__(self, name):
    """Handles all other method calls and does nothing."""
    return lambda *args, **kwargs: self


def get_monitoring_root() -> None:
  return None
