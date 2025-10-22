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

from typing import Callable

from absl import logging

_NO_FRAMEWORK = "NO_FRAMEWORK"
is_enabled: Callable[[], bool] = lambda: False
TraceAnnotation = None  # pylint: disable=invalid-name
start_server: Callable[[int], None] = lambda port: None
stop_server: Callable[[], None] = lambda: None
framework = _NO_FRAMEWORK

try:
  if framework == _NO_FRAMEWORK:
    from jax import profiler  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    TraceAnnotation = profiler.TraceAnnotation
    is_enabled = profiler.TraceAnnotation.is_enabled
    start_server = profiler.start_server
    stop_server = profiler.stop_server
    framework = "jax"
except ImportError:
  logging.warning("Failed to load jax profiler")


def is_loaded():
  return framework != _NO_FRAMEWORK
