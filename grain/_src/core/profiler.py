# Copyright 2025 Google LLC
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
"""Import wrapper for framework specific profilers."""

import functools
from typing import Callable

from absl import flags
from absl import logging

_GRAIN_ENABLE_MULTIPROCESS_WORKER_PROFILING = flags.DEFINE_bool(
    "grain_enable_multiprocess_worker_profiling",
    False,
    "If True, starts profiler servers on spawned worker processes to be"
    " profiled alongside the main process (when requested).",
)

# Internal constants.
_NO_FRAMEWORK = "NO_FRAMEWORK"
_framework = _NO_FRAMEWORK
_subprocess_hooks_loaded = False


class TraceAnnotation(object):
  """No-op trace annotation for when the profiler is not loaded."""

  def __init__(self, *args, **kwargs):
    del args, kwargs
    pass

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback
    pass


def is_enabled() -> bool:
  """Returns whether the profiler is enabled."""
  return False


def is_loaded() -> bool:
  """Returns whether the profiler is loaded."""
  return _framework != _NO_FRAMEWORK


def is_worker_profiling_supported() -> bool:
  """Returns whether worker profiling is supported."""
  return is_loaded() and _subprocess_hooks_loaded


def is_worker_profiling_enabled() -> bool:
  """Returns whether worker profiling is enabled."""
  return (
      is_worker_profiling_supported()
      and _GRAIN_ENABLE_MULTIPROCESS_WORKER_PROFILING.value
  )


def get_framework() -> str:
  """Returns the framework used for profiling."""
  return _framework


def start_server(port: int) -> None:
  """Starts the profiler server."""
  del port
  pass


def stop_server() -> None:
  """Stops the profiler server."""
  pass


def register_subprocess(pid: int, port: int) -> Callable[[], None]:
  """Registers a subprocess with its xprof port.

  Args:
    pid: The process ID of the subprocess.
    port: The port of the profiler server in the subprocess.

  Returns:
    A function that can be called to unregister the subprocess.

  Raises:
    RuntimeError: If the subprocess fails to be registered.
  """
  del pid, port
  raise RuntimeError(
      "Subprocess profiler registration is not supported in this environment."
  )


def get_worker_init_fn(port: int) -> Callable[[], None]:
  """Start the profiler server in a worker process."""
  if not is_worker_profiling_enabled():
    return lambda: None

  def _worker_init_fn() -> None:
    try:
      start_server(port)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Failed to start profiler server: %s", str(e))

  return _worker_init_fn


try:
  if _framework == _NO_FRAMEWORK:
    from jax import profiler  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    from jax._src.lib import jaxlib_extension_version  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    TraceAnnotation = profiler.TraceAnnotation
    is_enabled = profiler.TraceAnnotation.is_enabled
    # jaxlib_extension_version >=448 has subprocess profiling hooks enabled.
    if jaxlib_extension_version >= 448:
      # multiprocess workers will crash when attempting to connect to a backend
      # so we skip using requires_backend=False.
      start_server = functools.partial(
          profiler.start_server, requires_backend=False
      )
      register_subprocess = profiler.register_subprocess
      _subprocess_hooks_loaded = True
    else:
      start_server = profiler.start_server
      _subprocess_hooks_loaded = False
      logging.warning(
          "Grain multiprocess worker profiling requires jaxlib extension"
          " version 448 or later (jaxlib >= 0.11.0). Current version: %s.",
          jaxlib_extension_version,
      )

    stop_server = profiler.stop_server

    _framework = "jax"
except ImportError as e:
  logging.warning("Failed to load jax profiler: %s", e)
