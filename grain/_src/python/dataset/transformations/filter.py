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
      transform: Union[transforms.FilterTransform, Callable[[T], bool]],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.FilterTransform):
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


class _FilterDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that filters elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      filter_fn: Callable[[T], bool],
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._filter_fn = filter_fn

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
    if not passed_filter:
      raise StopIteration
    with self._stats.record_self_time(offset_ns=timer.value()):
      return self._stats.record_output_spec(value)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return f"FilterDatasetIterator(parent={self._parent}"


class FilterIterDataset(dataset.IterDataset[T]):
  """Filter transformation for IterDatasets."""

  _MUTATES_ELEMENT_SPEC = False

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: Union[transforms.FilterTransform, Callable[[T], bool]],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.FilterTransform):
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
        stats=self._stats,
    )

  def __str__(self) -> str:
    return f"FilterIterDataset(transform={self._transform_name})"
