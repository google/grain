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
"""This module contains implementations for various data sources.

Data source is an abstraction that is responsible for retrieving data records
from storage backend (e.g. a set of files, a database). It is used by the
DataTransformer to load data records. In V1 of the python backend, we focus on
data sources based on storage backends allowing efficient random access
(e.g. ArrayRecord files.) This allowshaving deterministic and preemtable input
pipelines.
"""

import collections
from collections.abc import Sequence
import math
from multiprocessing import shared_memory
import os
import threading
import time
import typing
from typing import Any, Generic, Optional, Protocol, SupportsIndex, TypeVar, Union

from absl import logging
import array_record.python.array_record_data_source as array_record
from etils import epath
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import usage_logging

from grain._src.core import monitoring  # pylint: disable=g-bad-import-order
from array_record.python.array_record_data_source import PathLikeOrFileInstruction

_api_usage_counter = monitoring.Counter(
    "/grain/python/data_sources/api",
    monitoring.Metadata(description="API initialization counter."),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)
_bytes_read_counter = monitoring.Counter(
    "/grain/python/data_sources/bytes_read",
    monitoring.Metadata(
        description=(
            "Number of bytes produced by a data source via random access."
        ),
    ),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("source", str)],
)

T = TypeVar("T")
ArrayRecordDataSourcePaths = Union[
    PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
]

_SparseArray = collections.namedtuple(
    "SparseArray", ["indices", "values", "dense_shape"]
)


class ArrayRecordDataSource(array_record.ArrayRecordDataSource):
  """Data source for ArrayRecord files."""

  def __init__(self, paths: ArrayRecordDataSourcePaths):
    """Creates a new ArrayRecordDataSource object.

    See `array_record.ArrayRecordDataSource` for more details.

    Args:
      paths: A single path/FileInstruction or list of paths/FileInstructions.
    """
    super().__init__(paths)
    _api_usage_counter.Increment("ArrayRecordDataSource")

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    data = super().__getitem__(record_key)
    _bytes_read_counter.IncrementBy(len(data), "ArrayRecordDataSource")
    return data

  @property
  def paths(self) -> ArrayRecordDataSourcePaths:
    return self._paths


@typing.runtime_checkable
class RandomAccessDataSource(Protocol, Generic[T]):
  """Interface for datasources where storage supports efficient random access.

  Note that `__repr__` has to be additionally implemented to make checkpointing
  work with this source.
  """

  def __len__(self) -> int:
    """Returns the total number of records in the data source."""

  def __getitem__(self, record_key: SupportsIndex) -> T:
    """Returns the value for the given record_key.

    This method must be threadsafe. It's also expected to be deterministic.
    When using multiprocessing (worker_count>0) PyGrain will pickle the data
    source, which invokes __getstate__(), and send a copy to each worker
    process, where __setstate__() is called. After that each worker process
    has its own independent data source object.

    Arguments:
      record_key: This will be an integer in [0, len(self)-1].

    Returns:
      The corresponding record. File data sources often return the raw bytes but
      records can be any Python object.
    """


class RangeDataSource:
  """Range data source, similar to python range() function."""

  def __init__(self, start: int, stop: int, step: int):
    assert step != 0, "step can't be zero."
    self._start = start
    self._stop = stop
    self._step = step
    self._len = int(math.ceil((self._stop - self._start) / step))
    assert self._len >= 0, "length can't be negative."
    _api_usage_counter.Increment("RangeDataSource")

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> int:
    record_key = record_key.__index__()
    if record_key < 0 or record_key >= self._len:
      raise IndexError(f"Index {record_key} out of range for {self._len}")
    return self._start + record_key * self._step

  def __repr__(self) -> str:
    return (
        f"RangeDataSource(start={self._start}, stop={self._stop}, "
        f"step={self._step})"
    )


class SharedMemoryDataSource(shared_memory.ShareableList):
  """Simple in-memory data source for sequences that is sharable among multiple processes.

  Note:
    This constrains storable values to only the int, float, bool, str (less than
    10M bytes each), bytes (less than 10M bytes each), and None built-in data
    types. It also notably differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert, etc.)
  """

  def __init__(
      self,
      elements: Optional[Sequence[Any]] = None,
      *,
      name: Optional[str] = None,
  ):
    """Creates a new InMemoryDataSource object.

    Args:
      elements: The elements for the sharable list.
      name: The name of the datasource.
    """
    if elements is not None:
      logging.info(
          "Creating a new ShareableList" + f" with name {name}"
          if name is not None
          else ""
      )
    elif name is not None:
      logging.info("Attaching to a ShareableList named %s", name)
    else:
      raise ValueError("Elements or name must be provided.")
    super().__init__(elements, name=name)
    _api_usage_counter.Increment("InMemoryDataSource")

  def __str__(self):
    return f"InMemoryDataSource(name={self.shm.name}, len={len(self)})"

  def close(self):
    self.shm.close()

  def unlink(self):
    self.shm.unlink()

  def __del__(self):
    del self.shm


# `tensor` can be a tf.Tensor, tf.SparseTensor or tf.RaggedTensor.
def _as_numpy(tensor):
  import tensorflow as tf  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  if isinstance(tensor, (tf.Tensor, tf.RaggedTensor)):
    return tensor.numpy()
  if isinstance(tensor, tf.SparseTensor):
    return _SparseArray(
        tensor.indices.numpy(),
        tensor.values.numpy(),
        tensor.dense_shape.numpy(),
    )
  raise ValueError(f"Type {type(tensor)} is not supported in PyGrain.")
