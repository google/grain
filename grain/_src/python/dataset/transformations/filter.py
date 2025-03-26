# Copyright 2023 Google LLC
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
"""Filter transformation for LazyDataset."""

import functools
from typing import Any, Callable, TypeVar, Union

from absl import logging
from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


class FilterMapDataset(dataset.MapDataset[T]):
  """Filter MapDataset."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.MapDataset[T],
      transform: Union[transforms.Filter, Callable[[T], bool]],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.Filter):
      self._filter_fn = transform.filter
      self._transform_cls_name = transform.__class__.__name__
    else:
      self._filter_fn = transform
      self._transform_cls_name = None

  @functools.cached_property
  def _transform_name(self):
    return self._transform_cls_name or transforms.get_pretty_transform_name(
        self._filter_fn
    )

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      if element is not None and self._filter_fn(element):
        return element
      return None

  def __str__(self) -> str:
    return f"FilterMapDataset(transform={self._transform_name})"


# The number of filtered elements is checked on intervals of this size.
_CHECK_FILTERED_INTERVAL = 1000
# The interval between warnings about filtered elements.
_WARN_FILTERED_INTERVAL_SEC = 60


class FilterThresholdChecker:
  """Warns or raises an error if the filtered elements ratio is too high."""

  def __init__(
      self,
      transform_name: str,
      warn_threshold: float | None,
      raise_threshold: float | None,
  ):
    self._transform_name = transform_name
    self._passed = 0
    self._skipped = 0
    self._warn_threshold: float = warn_threshold or float("inf")
    self._raise_threshold: float = raise_threshold or float("inf")

  def check(self, passed: bool) -> None:
    """Record whether an element was filtered out and check the ratio."""
    if passed:
      self._passed += 1
    else:
      self._skipped += 1

    if self._passed + self._skipped >= _CHECK_FILTERED_INTERVAL:
      skipped_ratio = self._skipped / (self._passed + self._skipped)
      if skipped_ratio >= self._raise_threshold:
        raise ValueError(
            f"Transformation {self._transform_name} skipped"
            f" {(skipped_ratio*100):.2f} % of the last seen"
            f" {self._passed + self._skipped} elements. Please check the"
            " filtering logic. If this is intended, consider pre-filtering the"
            " input dataset before training. To disable this check, set"
            " `grain.experimental.DatasetOptions.filter_raise_threshold_ratio`"
            " used in `WithOptionsIterDataset` to `None`."
        )
      if skipped_ratio >= self._warn_threshold:
        logging.log_every_n_seconds(
            logging.WARNING,
            "Transformation %s skipped %.2f %% of the last seen %d elements."
            " Please check the filtering logic. If this is intended, consider"
            " pre-filtering the input dataset before training. To disable this"
            " check, set"
            " `grain.experimental.DatasetOptions.filter_warn_threshold_ratio`"
            " used in `WithOptionsIterDataset` to `None`.",
            _WARN_FILTERED_INTERVAL_SEC,
            self._transform_name,
            skipped_ratio * 100,
            self._passed + self._skipped,
        )
      self._passed = 0
      self._skipped = 0


class _FilterDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that filters elements."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      filter_fn: Callable[[T], bool],
      transform_name: str,
  ):
    super().__init__(parent)
    self._filter_fn = filter_fn
    self._transform_name = transform_name

  @functools.cached_property
  def _threshold_checker(self):
    return FilterThresholdChecker(
        transform_name=str(self),
        warn_threshold=self._ctx.dataset_options.filter_warn_threshold_ratio,
        raise_threshold=self._ctx.dataset_options.filter_raise_threshold_ratio,
    )

  @dataset_stats.record_next_duration_if_output
  def __next__(self):
    value = None
    passed_filter = False
    timer = dataset_stats.Timer()
    while not passed_filter:
      try:
        value = next(self._parent)
      except StopIteration:
        break
      with timer:
        passed_filter = self._filter_fn(value)
        self._threshold_checker.check(passed_filter)
    if not passed_filter:
      raise StopIteration
    with self._stats.record_self_time(offset_ns=timer.value()):
      return self._stats.record_output_spec(value)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return f"FilterDatasetIterator(transform={self._transform_name})"


class FilterIterDataset(dataset.IterDataset[T]):
  """Filter transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: Union[transforms.Filter, Callable[[T], bool]],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.Filter):
      self._filter_fn = transform.filter
      self._transform_cls_name = transform.__class__.__name__
    else:
      self._filter_fn = transform
      self._transform_cls_name = None

  @functools.cached_property
  def _transform_name(self):
    return self._transform_cls_name or transforms.get_pretty_transform_name(
        self._filter_fn
    )

  def __iter__(self) -> _FilterDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _FilterDatasetIterator(
        parent_iter,
        filter_fn=self._filter_fn,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"FilterIterDataset(transform={self._transform_name})"
