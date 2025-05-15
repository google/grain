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
from __future__ import annotations

import functools
from typing import Sequence, Union

from absl import logging
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset


class SourceMapDataset(dataset.MapDataset):
  """Simple wrapper for random access data sources."""

  def __init__(self, source: base.RandomAccessDataSource):
    super().__init__()
    self._source = source

  def __len__(self) -> int:
    return len(self._source)

  def __str__(self) -> str:
    return f"SourceMapDataset(source={self._source.__class__.__name__})"

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      return self._stats.record_output_spec(self._source[index % len(self)])

  def log_lineage(self):
    pass

  @property
  def paths(self) -> str | Sequence[str]:
    if hasattr(self._source, "paths"):
      assert isinstance(self._source, base.RandomAccessDataSource)
      return self._source.paths
    else:
      return []


def log_lineage_for_sources(
    root: Union[dataset.MapDataset, dataset.IterDataset],
):
  """Traverses tree of transformations and logs lineage on source datasets."""
  pass


class RangeMapDataset(dataset.MapDataset[int]):
  """Range data source, similar to python range() function."""

  def __init__(self, start: int, stop: int | None = None, step: int = 1):
    super().__init__()
    self.start = 0 if stop is None else start
    self.stop = start if stop is None else stop
    self.step = step

  @functools.cached_property
  def _length(self) -> int:
    return len(range(self.start, self.stop, self.step))

  def __len__(self) -> int:
    return self._length

  def __str__(self) -> str:
    return (
        f"RangeMapDataset(start={self.start}, stop={self.stop},"
        f" step={self.step})"
    )

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      return self._stats.record_output_spec(
          self.start + (index % self._length) * self.step
      )

  def to_iter_dataset(
      self,
      read_options: options.ReadOptions | None = None,
      *,
      allow_nones: bool = False,
  ) -> dataset.IterDataset[int]:
    # Override the default multithreaded execution to avoid wasting memory.
    # The prefetch is not necessary since there's no IO.
    return super().to_iter_dataset(
        read_options=(
            read_options or options.ReadOptions(prefetch_buffer_size=0)
        ),
        allow_nones=allow_nones,
    )
