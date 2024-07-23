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

from typing import Union

from absl import logging
from grain._src.python.lazy_dataset import base
from grain._src.python.lazy_dataset import lazy_dataset


class SourceMapDataset(lazy_dataset.MapDataset):
  """Simple wrapper for random access data sources."""

  def __init__(self, source: base.RandomAccessDataSource):
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
    root: Union[lazy_dataset.MapDataset, lazy_dataset.IterDataset],
):
  """Traverses tree of transformations and logs lineage on source datasets."""
  pass
