"""Provides an `IterDataset` for Parquet file format."""

from typing import TypeVar

from etils import epy
from grain._src.python.dataset import dataset


# lazy import for pyarrow
with epy.lazy_imports():
  import pyarrow.parquet as pq  # pytype: disable=import-error # pylint: disable=g-import-not-at-top


T = TypeVar("T")


class _ParquetDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator for Parquet file format."""

  def __init__(
      self, path: str, row_group: int = 0, index_within_row_group: int = 0
  ):
    super().__init__()
    self._row_group = row_group
    self._index_within_row_group = index_within_row_group
    self._pq_path = path
    self._pq_file = pq.ParquetFile(self._pq_path)
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
