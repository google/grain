# Copyright 2022 Google LLC
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
"""Ragged batch transformation for LazyDataset."""

import math
from typing import Any, TypeVar

from grain._src.core.transforms import RaggedBatchTransform
from grain._src.python.lazy_dataset.lazy_dataset import LazyMapDataset


Element = Any
T = TypeVar("T")  # pylint: disable=invalid-name


class RaggedBatchLazyMapDataset(LazyMapDataset[T]):
  """Concatenate a fixed number of data records, which may be of different size."""

  def __init__(self, parent: LazyMapDataset, transform: RaggedBatchTransform):
    super().__init__()
    self._parent = parent
    self._transform = transform
    if parent.sparse:
      raise ValueError(
          f"Cannot create RaggedBatchLazyDataset for sparse parent {parent}."
      )

  @property
  def sparse(self) -> bool:
    return False

  def __len__(self) -> int:
    return math.ceil(len(self._parent) / self._transform.batch_size)

  def __getitem__(self, index: int) -> Element:
    indices = range(
        index * self._transform.batch_size,
        min(len(self._parent), (index + 1) * self._transform.batch_size),
    )
    return [self._parent[i] for i in indices]
