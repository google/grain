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
"""Map transformation for LazyDataset."""

import functools
from typing import Any, Callable, Protocol, Sequence, TypeVar, runtime_checkable

from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats


T = TypeVar("T")  # pylint: disable=invalid-name


@runtime_checkable
class SupportsElementSpecInference(Protocol):
  """A protocol for transformations supporting element spec inference."""

  def output_spec(self, input_spec: Any) -> Any:
    ...


class _ElementSpecFromTransformMapDatasetMixin(dataset.MapDataset[T]):
  """Mixin for element spec inference for `MapDataset`s accepting a transform.

  We have both `MapDataset` and `IterDataset` variants to avoid diamond
  inheritance pattern.
  """

  def __init__(self, parent: dataset.MapDataset, transform: Any):
    super().__init__(parent)
    self._transform = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._transform)

  @property
  def _element_spec(self) -> Any:
    if not isinstance(self._transform, SupportsElementSpecInference):
      raise ValueError(
          "Cannot infer element spec for transform"
          f" {self._transform_name} because it does not implement `output_spec`"
          f" method. Implement `{self._transform_name}.output_spec(self,"
          " input_spec: PyTree[ShapeDtypeStructProtocol]) -> "
          " PyTree[ShapeDtypeStructProtocol]` to enable element spec inference."
      )
    return self._transform.output_spec(self._parent._element_spec)  # pylint: disable=protected-access


class _ElementSpecFromTransformIterDatasetMixin(dataset.IterDataset[T]):
  """Mixin for element spec inference for `IterDataset`s accepting a transform.

  We have both `MapDataset` and `IterDataset` variants to avoid diamond
  inheritance pattern.
  """

  def __init__(self, parent: dataset.IterDataset, transform: Any):
    super().__init__(parent)
    self._transform = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._transform)

  @property
  def _element_spec(self) -> Any:
    if not isinstance(self._transform, SupportsElementSpecInference):
      raise ValueError(
          "Cannot infer element spec for transform"
          f" {self._transform_name} because it does not implement `output_spec`"
          f" method. Implement `{self._transform_name}.output_spec(self,"
          " input_spec: PyTree[ShapeDtypeStructProtocol]) -> "
          " PyTree[ShapeDtypeStructProtocol]` to enable element spec inference."
      )
    return self._transform.output_spec(self._parent._element_spec)  # pylint: disable=protected-access


class MapMapDataset(_ElementSpecFromTransformMapDatasetMixin[T]):
  """Map transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.Map | Callable[[Any], T],
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.Map):
      self._map_fn = transform.map
    else:
      self._map_fn = transform

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"MapMapDataset(transform={self._transform_name})"

  def _map_element(self, element: Any) -> T:
    if element is None:
      return None  # pyrefly: ignore[bad-return]
    return self._map_fn(element)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      mapped_element = self._map_element(element)
      return (
          self._stats.record_output_spec(mapped_element)
          if mapped_element is not None
          else None
      )

  def _getitems(self, indices: Sequence[int]):
    elements = self._parent._getitems(indices)  # pylint: disable=protected-access
    with self._stats.record_self_time(num_elements=len(indices)):
      processed_elements = [self._map_element(element) for element in elements]
      return self._stats.record_output_spec_for_batch(processed_elements)


class MapWithIndexMapDataset(_ElementSpecFromTransformMapDatasetMixin[T]):
  """Map with index transformation for MapDataset."""

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: transforms.MapWithIndex | Callable[[int, Any], T],
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.MapWithIndex):
      self._map_fn = transform.map_with_index
    else:
      self._map_fn = transform

  @functools.cached_property
  def _transform_name(self):
    return transforms.get_pretty_transform_name(self._map_fn)

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"MapWithIndexMapDataset(transform={self._transform_name})"

  def _map_with_index_fn(self, index: int, element: Any) -> T:
    if element is None:
      return None  # pyrefly: ignore[bad-return]
    return self._map_fn(index, element)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      mapped_element = self._map_with_index_fn(index, element)
      return (
          self._stats.record_output_spec(mapped_element)
          if mapped_element is not None
          else None
      )

  def _getitems(self, indices: Sequence[int]):
    elements = self._parent._getitems(indices)  # pylint: disable=protected-access
    with self._stats.record_self_time(num_elements=len(indices)):
      processed_elements = [
          self._map_with_index_fn(index, element)
          for index, element in zip(indices, elements)
      ]
    return self._stats.record_output_spec_for_batch(processed_elements)


class _MapDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies map transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      map_fn: Callable[[Any], T],
      transform_name: str,
  ):
    super().__init__(parent)
    self._map_fn = map_fn
    self._transform_name = transform_name

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        element = self._map_fn(element)
      return self._stats.record_output_spec(element)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)

  def _get_next_index(self) -> int:
    return dataset.get_next_index(self._parent)

  def _set_next_index(self, next_index: int):
    dataset.set_next_index(self._parent, next_index)

  def __str__(self) -> str:
    return f"MapDatasetIterator(transform={self._transform_name})"


class _MapWithIndexDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that applies map with index transformation to elements."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      map_fn: Callable[[int, Any], T],
      transform_name: str,
  ):
    super().__init__(parent)
    self._map_fn = map_fn
    self._transform_name = transform_name
    self._counter = 0

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_PREPROCESSING)
  def __next__(self):
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is not None:
        element = self._map_fn(self._counter, element)
      self._counter += 1
      return self._stats.record_output_spec(element)

  def get_state(self):
    return {
        "parent": self._parent.get_state(),
        "counter": self._counter,
    }

  def set_state(self, state):
    self._parent.set_state(state["parent"])
    self._counter = state["counter"]

  def _get_next_index(self) -> int:
    return self._counter

  def _set_next_index(self, next_index: int):
    dataset.set_next_index(self._parent, next_index)
    self._counter = next_index

  def __str__(self) -> str:
    return f"MapWithIndexDatasetIterator(transform={self._transform_name})"


class MapIterDataset(_ElementSpecFromTransformIterDatasetMixin[T]):
  """Map transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.Map | Callable[[Any], T],
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.Map):
      self._map_fn = transform.map
    else:
      self._map_fn = transform

  def __iter__(self) -> _MapDatasetIterator[T]:
    return _MapDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"MapIterDataset(transform={self._transform_name})"


class MapWithIndexIterDataset(_ElementSpecFromTransformIterDatasetMixin[T]):
  """Map with index transformation for IterDatasets."""

  def __init__(
      self,
      parent: dataset.IterDataset,
      transform: transforms.MapWithIndex | Callable[[int, Any], T],
  ):
    super().__init__(parent, transform)
    if isinstance(transform, transforms.MapWithIndex):
      self._map_fn = transform.map_with_index
    else:
      self._map_fn = transform

  def __iter__(self) -> _MapWithIndexDatasetIterator[T]:
    return _MapWithIndexDatasetIterator(
        self._parent.__iter__(),
        map_fn=self._map_fn,
        transform_name=self._transform_name,
    )

  def __str__(self) -> str:
    return f"MapWithIndexIterDataset(transform={self._transform_name})"
