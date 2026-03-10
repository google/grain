# Copyright 2025 Google LLC
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
"""Provides an `IterDataset` for Parquet file format."""

from typing import IO, Sequence, TypeVar, Union

from etils import epy
from grain._src.python.dataset import dataset


# lazy import for pyarrow
with epy.lazy_imports():
  import pyarrow.parquet as pq  # pytype: disable=import-error # pylint: disable=g-import-not-at-top


T = TypeVar("T")

ParquetDataSourcePath = Union[str, IO[bytes]]
ParquetDataSource = Union[
    ParquetDataSourcePath, Sequence[ParquetDataSourcePath]
]


class _ParquetDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator for Parquet file format."""

  def __init__(
      self,
      sources: Sequence[ParquetDataSourcePath],
      row_group: int = 0,
      index_within_row_group: int = 0,
      source_index: int = 0,
      **read_kwargs,
  ):
    super().__init__()
    self._sources = sources
    self._source_index = source_index
    self._row_group = row_group
    self._index_within_row_group = index_within_row_group
    self._read_kwargs = read_kwargs
    self._open_current_source()

  def _open_current_source(self):
    self._pq_file = pq.ParquetFile(
        self._sources[self._source_index], **self._read_kwargs
    )
    self._read_row_group_to_np_table()

  def _read_row_group_to_np_table(self):
    table = self._pq_file.read_row_group(self._row_group)
    self._row_group_len = len(table)
    self._np_table = {}
    for i in range(table.num_columns):
      self._np_table[table.field(i).name] = table.column(i).to_numpy()

  def __next__(self):
    if self._index_within_row_group >= self._row_group_len:
      if self._row_group < self._pq_file.num_row_groups - 1:
        self._row_group += 1
        self._index_within_row_group = 0
        self._read_row_group_to_np_table()
        return self.__next__()
      elif self._source_index < len(self._sources) - 1:
        self._source_index += 1
        self._row_group = 0
        self._index_within_row_group = 0
        self._open_current_source()
        return self.__next__()
      else:
        raise StopIteration()
    else:
      item = {
          k: v[self._index_within_row_group] for k, v in self._np_table.items()
      }
      self._index_within_row_group += 1
      return item

  def get_state(self):
    return {
        "source_index": self._source_index,
        "row_group": self._row_group,
        "index_within_row_group": self._index_within_row_group,
    }

  def set_state(self, state):
    self._source_index = state["source_index"]
    self._row_group = state["row_group"]
    self._index_within_row_group = state["index_within_row_group"]
    self._open_current_source()


class ParquetIterDataset(dataset.IterDataset[T]):
  """An IterDataset for a parquet format file."""

  def __init__(self, path: ParquetDataSource, **read_kwargs):
    """Initializes ParquetIterDataset.

    Args:
      path: A path or sequence of paths to parquet format files.
      **read_kwargs: Keyword arguments to pass to pyarrow.parquet.ParquetFile.
    """
    super().__init__()
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
      self._paths = [path]
    else:
      self._paths = path
    self._read_kwargs = read_kwargs

  def __iter__(self) -> _ParquetDatasetIterator[T]:
    return _ParquetDatasetIterator(self._paths, **self._read_kwargs)

  def set_slice(self, sl: slice, sequential_slice: bool = False):
    del sequential_slice
    self._paths = self._paths[sl]
