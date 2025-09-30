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

from typing import Sequence, Union

from absl import logging
from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats


class SourceMapDataset(dataset.MapDataset):
  """Simple wrapper for random access data sources."""

  def __init__(self, source: base.RandomAccessDataSource):
    super().__init__()
    self._source = source
    self._original_source_map_dataset = None

  def __len__(self) -> int:
    return len(self._source)

  def __str__(self) -> str:
    return f"SourceMapDataset(source={self._source.__class__.__name__})"

  @dataset_stats.trace_input_pipeline(stage_category=dataset_stats.IPL_CAT_READ)
  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    with self._stats.record_self_time():
      return self._stats.record_output_spec(self._source[index % len(self)])

  def _getitems(self, indices: Sequence[int]):
    if not isinstance(
        self._source, base.SupportsBatchedReadRandomAccessDataSource
    ):
      return super()._getitems(indices)
    return self._source._getitems([index % len(self) for index in indices])  # pylint: disable=protected-access

  def _get_sequential_slice(self, sl: slice) -> slice:
    """Returns the sequential slice per worker."""
    worker_index = sl.start
    workers_count = sl.step
    shard_options = sharding.ShardOptions(
        shard_index=worker_index,
        shard_count=workers_count,
        drop_remainder=False,
    )
    shard_start, shard_end = sharding.even_split(self.__len__(), shard_options)
    return slice(shard_start, shard_end)

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    assert sequential_slice, "Only sequential slicing is supported."
    if not self._original_source_map_dataset:
      self._original_source_map_dataset = SourceMapDataset(self._source)
    new_slice = self._get_sequential_slice(sl)
    self._source = self._original_source_map_dataset.slice(new_slice)

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
    self.original_start = self.start
    self.original_len = len(range(self.start, self.stop, self.step))
    self._length = self.original_len

  def __len__(self) -> int:
    return self._length

  def __str__(self) -> str:
    return (
        f"RangeMapDataset(start={self.start}, stop={self.stop},"
        f" step={self.step})"
    )

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    assert (
        sequential_slice
    ), "Only sequential slicing is supported for RangeMapDataset."
    worker_index = sl.start
    workers_count = sl.step
    shard_options = sharding.ShardOptions(
        shard_index=worker_index,
        shard_count=workers_count,
        drop_remainder=False,
    )
    shard_start, shard_end = sharding.even_split(
        self.original_len, shard_options
    )
    self.start = self.original_start + shard_start * self.step
    self.stop = self.original_start + shard_end * self.step
    self._length = len(range(self.start, self.stop, self.step))

  @dataset_stats.trace_input_pipeline(stage_category=dataset_stats.IPL_CAT_READ)
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
