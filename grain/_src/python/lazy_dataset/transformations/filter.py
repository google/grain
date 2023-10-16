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

from typing import Any, Callable, TypeVar

from grain._src.core import transforms
from grain._src.python.lazy_dataset import lazy_dataset


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


@lazy_dataset.lazy_map_dataset_function("filter")
class FilterLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Filter LazyMapDataset."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset[T],
      transform: transforms.FilterTransform | Callable[[T], bool],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.FilterTransform):
      self._filter_fn = transform.filter
    else:
      self._filter_fn = transform

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    if element is None:
      return None
    if self._filter_fn(element):
      return element
    return None

  def __str__(self) -> str:
    return f"FilterLazyMapDataset(parent={self._parent})"


class _FilterLazyDatasetIterator(lazy_dataset.LazyDatasetIterator[T]):
  """Iterator that filters elements."""

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
      filter_fn: Callable[[T], bool],
  ):
    super().__init__()
    self._parent = parent
    self._filter_fn = filter_fn

  def __next__(self):
    value = None
    filtered_value = False
    while not filtered_value:
      try:
        value = next(self._parent)
      except StopIteration:
        break
      filtered_value = self._filter_fn(value)
    if not filtered_value:
      raise StopIteration
    return value

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def __str__(self) -> str:
    return f"FilterLazyDatasetIterator(parent={self._parent}"


@lazy_dataset.lazy_iter_dataset_function("filter")
class FilterLazyIterDataset(lazy_dataset.LazyIterDataset[T]):
  """Filter transformation for LazyIterDatasets."""

  def __init__(
      self,
      parent: lazy_dataset.LazyIterDataset,
      transform: transforms.FilterTransform | Callable[[T], bool],
  ):
    super().__init__(parent)
    if isinstance(transform, transforms.FilterTransform):
      self._filter_fn = transform.filter
    else:
      self._filter_fn = transform

  def __iter__(self) -> _FilterLazyDatasetIterator[T]:
    parent_iter = self._parent.__iter__()
    return _FilterLazyDatasetIterator(
        parent_iter,
        filter_fn=self._filter_fn,
    )

  def __str__(self) -> str:
    return f"FilterLazyIterDataset(parent={self._parent}"
