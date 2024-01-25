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
"""LazyDataset data sources."""

from typing import Protocol

from absl import logging
from grain._src.python.lazy_dataset import lazy_dataset


class RandomAccessDataSource(Protocol):
  """Interface for datasets where storage supports efficient random access."""

  def __len__(self):
    ...

  def __getitem__(self, index: int):
    ...


class SourceLazyMapDataset(lazy_dataset.LazyMapDataset):
  """Simple wrapper for random access data sources."""

  _source: RandomAccessDataSource

  def __init__(self, source: RandomAccessDataSource):
    super().__init__()
    self._source = source

  def __len__(self) -> int:
    return len(self._source)

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    return self._source[index % len(self)]

  def log_lineage(self):
    pass


def log_lineage_for_sources(
    root: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
):
  """Traverses tree of transformations and logs lineage on source datasets."""
  pass
