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
"""This module contains implementations for the continual data source.

This data source works in conjuntion with the continual sequence sampler to
retrieve clips sequentially from within an element.
"""
from typing import Generic, SupportsIndex, Tuple, TypeVar

from grain._src.python import data_sources
from grain._src.python.experimental.continual_sequence_sampler import continual_sequence_sampler

T = TypeVar("T")


class ContinualDataSource(Generic[T]):
  """Data source for continual sequence sampler of data elements."""

  def __init__(
      self,
      element_source: data_sources.RandomAccessDataSource[T],
      sampler: continual_sequence_sampler.SamplerWrapper,
  ):
    self._element_source = element_source
    self._sampler = sampler

  def __len__(self) -> int:
    return self._sampler.length()

  def __getitem__(self, record_key: SupportsIndex) -> Tuple[T, int]:
    record_key = record_key.__index__()
    element_clip = self._sampler.record_key_to_element_and_clip(record_key)
    element = self._element_source[element_clip.element]
    return element, element_clip.clip
