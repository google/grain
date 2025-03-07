"""Provides an `IterDataset` for Parquet file format."""

from typing import TypeVar
from grain._src.python.dataset import dataset
import pyarrow.parquet as pq


T = TypeVar("T")


class _ParquetDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator for Parquet file format."""

  def __init__(
      self, path: str, row_group: int = 0, index_within_row_group: int = 0
  ):
    super().__init__()
    self._row_group = row_group
    self._index_within_row_group = index_within_row_group
    self._pq_file = pq.ParquetFile(path)
    self._table = self._pq_file.read_row_group(self._row_group)
    self._row_group_len = len(self._table)
    self._num_row_groups = self._pq_file.num_row_groups

  def __next__(self):
    if self._index_within_row_group >= self._row_group_len:
      if self._row_group < self._num_row_groups - 1:
        self._row_group += 1
        self._index_within_row_group = 0
        self._table = self._pq_file.read_row_group(self._row_group)
        self._row_group_len = len(self._table)
        return self.__next__()
      else:
        raise StopIteration()
    else:
      item = self._table.to_pylist()[self._index_within_row_group]
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
    self._table = self._pq_file.read_row_group(self._row_group)
    self._row_group_len = len(self._table)


class ParquetIterDataset(dataset.IterDataset[T]):
  """An IterDataset for a parquet format file."""

  def __init__(self, path: str):
    """Initializes ParquetIterDataset.

    Args:
      path: A path to a record io format file.
    """
    super().__init__()
    self._path = path

  def __iter__(self) -> _ParquetDatasetIterator[T]:
    return _ParquetDatasetIterator(self._path)
