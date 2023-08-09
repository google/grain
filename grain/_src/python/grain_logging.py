"""A logging module that allows setting a prefix applied to all log statements.

Usage:

from grain._src.python import grain_logging

grain_logging.set_prefix("foo")
grain_logging.info("My log message %i", 42)

will produce
[foo]: My log message 42
instead of
My log message 42


Log functions (info(...), error(...), etc.) are thread-safe, but
grain_logging.set_prefix(...) should be synchronized (most likely, it should be
called just once before the first log statement, but changing the prefix is
allowed).

It works as expected with "spawn"ing processes, but may be tricky if you
fork (you won't get an error when trying to log before setting the prefix
since the parent's prefix will be used).
"""

import logging
from typing import Any, ClassVar


class _PrefixLoggerAdapter(logging.LoggerAdapter):

  def process(self, msg, kwargs):
    return "[%s]: %s" % (self.extra["prefix"], msg), kwargs


class _GlobalLoggerAdapter:
  """A class holding the global LoggerAdapter."""

  _logger: ClassVar[logging.LoggerAdapter | None] = None

  @classmethod
  def set_prefix(cls, prefix: str) -> None:
    cls._logger = _PrefixLoggerAdapter(
        logging.getLogger("grain"), {"prefix": prefix}
    )

  @classmethod
  def get_logger(cls) -> logging.LoggerAdapter:
    if cls._logger is None:
      raise RuntimeError(
          "Trying to log with grain_logging.log before setting the prefix. "
          "Please call grain_logging.set_prefix(...) first."
      )
    return cls._logger


################################################################################
### Public API
################################################################################
def set_prefix(prefix: str) -> None:
  _GlobalLoggerAdapter.set_prefix(prefix)


def error(msg: Any, *args: Any, **kwargs: Any) -> None:
  _GlobalLoggerAdapter.get_logger().error(msg, *args, **kwargs)


def warning(msg: Any, *args: Any, **kwargs: Any) -> None:
  _GlobalLoggerAdapter.get_logger().warning(msg, *args, **kwargs)


def info(msg: Any, *args: Any, **kwargs: Any) -> None:
  _GlobalLoggerAdapter.get_logger().info(msg, *args, **kwargs)


def debug(msg: Any, *args: Any, **kwargs: Any) -> None:
  _GlobalLoggerAdapter.get_logger().debug(msg, *args, **kwargs)


def exception(msg: Any, *args: Any, **kwargs: Any) -> None:
  _GlobalLoggerAdapter.get_logger().exception(msg, *args, **kwargs)
