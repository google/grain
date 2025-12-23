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
"""Iterator that filters the stacktrace of thrown errors."""

from __future__ import annotations

from typing import Any, TypeVar

from grain._src.core import traceback_util
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats


traceback_util.register_exclusion(__file__)


T = TypeVar("T")


class TracebackFilterDatasetIterator(dataset.DatasetIterator[T]):
  """Filters internal stack frames from the stacktrace of thrown errors."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self, parent: dataset.DatasetIterator[T], traceback_filter_mode: str
  ):
    super().__init__(parent)
    self._traceback_filter_mode = traceback_filter_mode

  @traceback_util.run_with_traceback_filter
  @dataset_stats.record_next_duration_if_output
  @dataset_stats.trace_input_pipeline_next(
      stage_category=dataset_stats.IPL_CAT_META
  )
  def __next__(self) -> T:
    element = next(self._parent)
    with self._stats.record_self_time():
      return self._stats.record_output_spec(element)

  def get_state(self) -> dict[str, Any]:
    return self._parent.get_state()

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return f"TracebackFilterDatasetIterator(traceback_filter_mode={self._traceback_filter_mode})"
