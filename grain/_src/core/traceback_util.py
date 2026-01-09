# Copyright 2020 The JAX Authors.
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
"""Utility for filtering stack traces for readability.

Copied and modified from jax-ml/jax/jax/_src/traceback_util.py
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import pathlib
import traceback
import types
from typing import Any, TypeVar, cast

from grain._src.core.config import config  # pylint: disable=g-importing-member

C = TypeVar("C", bound=Callable[..., Any])

_grain_message_append = (
    "The stack trace below excludes Grain-internal frames.\n"
    "The preceding is the original exception that occurred, unmodified.\n"
    "\n--------------------"
)


def include_frame(f: types.FrameType) -> bool:
  return include_filename(f.f_code.co_filename)


def include_filename(filename: str) -> bool:
  # We want to exclude all files in `grain/_src` and its subdirectories.
  # Pathlib ensures path separator differences are accounted for on
  # different platforms.
  try:
    parts = pathlib.Path(filename).parts
  except Exception:  # pylint: disable=broad-except
    return True
  for i in range(len(parts) - 1):
    if parts[i] == "grain" and parts[i + 1] == "_src":
      return False
  return True


def _add_tracebackhide_to_hidden_frames(tb: types.TracebackType):
  for f, _ in traceback.walk_tb(tb):
    if not include_frame(f) and not _is_reraiser_frame(f):
      f.f_locals["__tracebackhide__"] = True


def filter_traceback(tb: types.TracebackType) -> types.TracebackType | None:
  out = None
  # Scan the traceback and collect relevant frames.
  frames = list(traceback.walk_tb(tb))
  for f, lineno in reversed(frames):
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)
  return out


def _add_call_stack_frames(tb: types.TracebackType) -> types.TracebackType:
  """Continues up the call stack and adds relevant frames to the traceback.

  We would like to avoid stepping too far up, e.g. past the exec/eval point of
  a REPL such as IPython. To that end, we stop past the first contiguous bunch
  of module-level frames, if we reach any such frames at all. This is a
  heuristic that might stop in advance of the REPL boundary. For example, if
  the call stack includes module-level frames from the current module A, and
  the current module A was imported from within a function F elsewhere, then
  the stack trace we produce will be truncated at F's frame.

  Args:
    tb: The current traceback.

  Returns:
    A new traceback including frames from the call stack.
  """
  out = tb

  reached_module_level = False
  for f, lineno in traceback.walk_stack(tb.tb_frame):
    if reached_module_level and f.f_code.co_name != "<module>":
      break
    if include_frame(f):
      out = types.TracebackType(out, f, f.f_lasti, lineno)
    if f.f_code.co_name == "<module>":
      reached_module_level = True
  return out


def _is_reraiser_frame(f: traceback.FrameSummary | types.FrameType) -> bool:
  if isinstance(f, traceback.FrameSummary):
    filename, name = f.filename, f.name
  else:
    filename, name = f.f_code.co_filename, f.f_code.co_name
  return filename == __file__ and name == "reraise_with_filtered_traceback"


def _is_under_reraiser(e: BaseException) -> bool:
  if e.__traceback__ is None:
    return False
  tb = traceback.extract_stack(e.__traceback__.tb_frame)
  return any(_is_reraiser_frame(f) for f in tb[:-1])


def format_exception_only(e: BaseException) -> str:
  return "".join(traceback.format_exception_only(type(e), e)).strip()


class UnfilteredStackTraceError(Exception):
  pass


_simplified_tb_msg = (
    "For simplicity, Grain has removed its internal frames "
    "from the traceback of the following exception. Set "
    "--grain_py_traceback_filtering=off to include these."
)


class SimplifiedTracebackError(Exception):
  def __str__(self):
    return _simplified_tb_msg


SimplifiedTracebackError.__module__ = "grain.errors"


def _running_under_ipython() -> bool:
  """Returns true if we appear to be in an IPython session."""
  try:
    get_ipython()  # pylint: disable=undefined-variable # type: ignore
    return True
  except NameError:
    return False


def _ipython_supports_tracebackhide() -> bool:
  """Returns true if the IPython version supports __tracebackhide__."""
  import IPython  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  return IPython.version_info[:2] >= (7, 17)


def _filtering_mode() -> str:
  mode = config.get_or_default("py_traceback_filtering")
  if mode == "auto":
    if (_running_under_ipython() and _ipython_supports_tracebackhide()):
      mode = "tracebackhide"
    else:
      mode = "quiet_remove_frames"
  return mode


def run_with_traceback_filter(fun: C) -> C:
  """Wraps ``fun`` to form a boundary for filtering exception tracebacks.

  When an exception occurs below ``fun``, this appends to it a custom
  ``__cause__`` that carries a filtered traceback. The traceback imitates the
  stack trace of the original exception, but with JAX-internal frames removed.

  This boundary annotation works in composition with itself. The topmost frame
  corresponding to an :func:`~api_boundary` is the one below which stack traces
  are filtered. In other words, if ``api_boundary(f)`` calls
  ``api_boundary(g)``, directly or indirectly, the filtered stack trace provided
  is the same as if ``api_boundary(f)`` were to simply call ``g`` instead.

  This annotation is primarily useful in wrapping functions output by Grain's
  transformations. For example, consider ``ds = ds.map(f)`` which returns a new
  dataset. Upon evaluation, the dataset recursively calls map on the parent
  dataset transformation upon which it was constructed from. If the function
  ``f`` raises an exception, the stack unwinds recursively through each
  transformation's callstack up to the original call site of ``g``.
  Because the function returned by :func:`~dataset.map` is annotated as an
  :func:`~api_boundary`, such an exception is accompanied by an additional
  traceback that excludes the frames specific to Grain's implementation.

  Summarizing the different modes:
   - "off": No filtering is applied.
   - "auto": The default mode. If running under IPython and the IPython version
     supports the __tracebackhide__ feature, "tracebackhide" is used. Otherwise,
     "quiet_remove_frames" is used.
   - "tracebackhide": The frames are marked with __tracebackhide__.
   - "quiet_remove_frames": The frames are removed, but a note is added to the
     exception message.
   - "remove_frames": The frames are removed, and the original exception is
     included as the cause of a new exception.

  Args:
    fun: The function to wrap.

  Returns:
    A wrapped version of `fun` that filters tracebacks.
  """

  # Short circuit if the function is already filtered to avoid wrapping
  # the same function multiple times.
  if getattr(fun, "is_traceback_filtered", False):
    return fun

  @functools.wraps(fun)
  def reraise_with_filtered_traceback(*args, **kwargs):
    __tracebackhide__ = True  # pylint: disable=invalid-name,unused-variable
    try:
      return fun(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      mode = _filtering_mode()
      if _is_under_reraiser(e) or mode == "off":
        raise
      if mode == "tracebackhide":
        _add_tracebackhide_to_hidden_frames(e.__traceback__)
        raise

      tb = e.__traceback__
      try:
        e.with_traceback(filter_traceback(tb))
        if mode == "quiet_remove_frames":
          e.add_note("--------------------\n" + _simplified_tb_msg)
        else:
          if mode == "remove_frames":
            msg = format_exception_only(e)
            msg = f"{msg}\n\n{_grain_message_append}"
            unfiltered_error = UnfilteredStackTraceError(msg)
            unfiltered_error.with_traceback(_add_call_stack_frames(tb))
          else:
            raise ValueError(
                f"grain_py_traceback_filtering={mode} is not a valid value."
            ) from e
          unfiltered_error.__cause__ = e.__cause__
          unfiltered_error.__context__ = e.__context__
          unfiltered_error.__suppress_context__ = e.__suppress_context__
          e.__cause__ = unfiltered_error
          e.__context__ = None
          del unfiltered_error
        raise
      finally:
        del mode, tb
  reraise_with_filtered_traceback.is_traceback_filtered = True
  return cast(C, reraise_with_filtered_traceback)
