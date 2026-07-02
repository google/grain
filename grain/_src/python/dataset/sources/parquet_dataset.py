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

from typing import Sequence, TypeVar

from etils import epy
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import interleave


# lazy import for pyarrow
with epy.lazy_imports():
  import pyarrow.parquet as pq  # pytype: disable=import-error # pylint: disable=g-import-not-at-top


T = TypeVar("T")

ParquetDataSourcePath = str | Sequence[str]
_CYCLE_LENGTH = 16


class _ParquetDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator for Parquet file format."""

  def __init__(
      self,
      path: str,
      row_group: int = 0,
      index_within_row_group: int = 0,
      **read_kwargs,
  ):
    super().__init__()
    self._row_group = row_group
    self._index_within_row_group = index_within_row_group
    self._pq_path = path
    self._pq_file = pq.ParquetFile(self._pq_path, **read_kwargs)
    self._np_table = {}
    self._row_group_len = 0
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
        "row_group": self._row_group,
        "index_within_row_group": self._index_within_row_group,
    }

  def set_state(self, state):
    self._row_group = state["row_group"]
    self._index_within_row_group = state["index_within_row_group"]
    self._read_row_group_to_np_table()


class ParquetIterDataset(dataset.IterDataset[T]):
  """An IterDataset for a Parquet format file.

  This dataset provides an iterator over records stored in Parquet files. It
  natively handles both single-file reads and multi-file interleaving. If a
  sequence of multiple paths is provided, the dataset automatically interleaves
  reads from the files (reading up to 16 files concurrently by default).

  Additional keyword arguments provided during initialization are forwarded
  directly to the underlying `pyarrow.parquet.ParquetFile` constructor. This
  allows users to configure advanced Arrow features like memory mapping or
  custom buffer sizes.

  Example:
    Initializing a dataset to read records from a Parquet file with `memory_map`
    option passed to `ParquetFile`::

      import tempfile
      import grain
      import pyarrow as pa
      import pyarrow.parquet as pq

      with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        table = pa.table({"id": [1, 2], "val": ["A", "B"]})
        pq.write_table(table, tmp.name)

        # Create a Parquet dataset with a keyword arg.
        ds = grain.experimental.ParquetIterDataset(
            tmp.name,
            memory_map=True
        )

        # Print each record from the dataset.
        for record in ds:
          print(record)
  """

  def __init__(
      self,
      path: ParquetDataSourcePath,
      **read_kwargs,
  ):
    """Initializes ParquetIterDataset.

    Args:
      path: A path or sequence of paths to Parquet format files. If multiple
        paths are provided, they are interleaved with at most 16 files read
        concurrently.
      **read_kwargs: Keyword arguments to pass to `pyarrow.parquet.ParquetFile`.
    """
    super().__init__()
    if isinstance(path, (str, bytes)):
      self._paths = [path]
    else:
      self._paths = list(path)
    self._read_kwargs = read_kwargs

  def __iter__(self) -> dataset.DatasetIterator[T]:
    if len(self._paths) == 1:
      return _ParquetDatasetIterator(self._paths[0], **self._read_kwargs)

    datasets = [ParquetIterDataset(p, **self._read_kwargs) for p in self._paths]
    delegate = interleave.InterleaveIterDataset(
        datasets, cycle_length=_CYCLE_LENGTH
    )
    return delegate.__iter__()

  def set_slice(self, sl: slice, sequential_slice: bool = False):
    del sequential_slice
    self._paths = self._paths[sl]
