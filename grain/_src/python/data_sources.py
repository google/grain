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

from collections.abc import Sequence
import inspect
import math
from multiprocessing import shared_memory
import os
import threading
import time
from typing import Any, Generic, Optional, SupportsIndex, TypeVar, Union

from absl import logging
from etils import epath
from grain._src.core import monitoring
from grain._src.python.dataset import stats as dataset_stats


# pylint: disable=g-import-not-at-top, g-importing-member, g-bad-import-order
import platform
if platform.system() == "Windows":
  PathLikeOrFileInstruction = Any
  class ARDataSource:
    def __init__(self, *args, **kwargs):
      raise RuntimeError("array_record isn't supported on Windows")
else:
  from array_record.python.array_record_data_source import (
      ArrayRecordDataSource as ARDataSource,
      PathLikeOrFileInstruction,
  )
# pylint: enable=g-import-not-at-top, g-importing-member, g-bad-import-order

_api_usage_counter = monitoring.Counter(
    "/grain/python/data_sources/api",
    monitoring.Metadata(description="API initialization counter."),
    fields=[("name", str)],
)

T = TypeVar("T")
ArrayRecordDataSourcePaths = Union[
    PathLikeOrFileInstruction, Sequence[PathLikeOrFileInstruction]
]

ArrayRecordReaderOptions = dict[str, str] | None


class ArrayRecordDataSource(ARDataSource):
  """Data source for ArrayRecord files.

  Reads serialized byte records from one or more ArrayRecord files or file
  instructions with random access indexing support.

  Example:
    Writing to and reading records from an ArrayRecord file using MapDataset::

      from array_record.python import array_record.python.array_record_data_source as array_record_module
      import grain

      # Write records to an ArrayRecord file.
      writer = array_record_module.ArrayRecordWriter(
          "/tmp/data.array_record", "group_size:1"
      )
      writer.write(b"example_byte_record_1")
      writer.write(b"example_byte_record_2")
      writer.close()

      # Load the ArrayRecord file using ArrayRecordDataSource.
      source = grain.sources.ArrayRecordDataSource(
          ["/tmp/data.array_record"]
      )
      parent_ds = grain.MapDataset.source(source)
      print(len(parent_ds))
      # 2
      print(parent_ds[0])
      # b'example_byte_record_1'
  """

  def __init__(
      self,
      paths: ArrayRecordDataSourcePaths,
      reader_options: ArrayRecordReaderOptions = None,
  ):
    """Creates a new ArrayRecordDataSource object.

    See `array_record.ArrayRecordDataSource` for more details.

    Args:
      paths: A single path/FileInstruction or list of paths/FileInstructions.
      reader_options: a dict[str, str] to be passed when creating a reader. For
        example, {index_storage_option:"in_memory"} stores the reader indices in
        memory versus {index_storage_option:"offloaded"} stores the indices on
        disk to save memory usage.

    Raises:
      ValueError: If `reader_options` is provided but not supported by the
        underlying `ArrayRecord` reader version.
    """
    array_record_signature = inspect.signature(ARDataSource.__init__)
    if "reader_options" in array_record_signature.parameters:
      super().__init__(paths, reader_options)
    elif reader_options is not None:
      # Reader options should not be set if they are not supported by the
      # current version of ArrayRecord.
      raise ValueError(
          "reader_options is not supported in this version of ArrayRecord."
      )
    else:
      super().__init__(paths)
    _api_usage_counter.Increment("ArrayRecordDataSource")

  @dataset_stats.trace_input_pipeline(stage_category=dataset_stats.IPL_CAT_READ)
  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    return super().__getitem__(record_key)

  @property
  def paths(self) -> ArrayRecordDataSourcePaths:
    return self._paths


class RangeDataSource:
  """Range data source, similar to python range() function.

  Produces a sequence of integers from `start` to `stop` with a given `step`,
  supporting efficient indexing and length lookup without loading data into
  memory.

  Example:
    Creating a range data source and accessing elements::

      import grain

      # Create a range data source
      source = grain.sources.RangeDataSource(start=0, stop=10, step=2)

      # Create a MapDataset from the source
      ds = grain.MapDataset.source(source)

      # Print the length of the dataset
      print(len(ds))
      # 5

      # Print all elements in the dataset
      print(list(ds))
      # [0, 2, 4, 6, 8]
  """

  def __init__(self, start: int, stop: int, step: int):
    """Initializes the RangeDataSource.

    Args:
      start: The start value of the range sequence.
      stop: The stop boundary of the range sequence (exclusive).
      step: The step increment between consecutive elements. Must not be 0.

    Raises:
      AssertionError: If `step` is 0 or if computed length is negative.
    """
    assert step != 0, "step can't be zero."
    self._start = start
    self._stop = stop
    self._step = step
    self._len = int(math.ceil((self._stop - self._start) / step))
    assert self._len >= 0, "length can't be negative."
    _api_usage_counter.Increment("RangeDataSource")

  def __len__(self) -> int:
    return self._len

  @dataset_stats.trace_input_pipeline(stage_category=dataset_stats.IPL_CAT_READ)
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

  Example:
    Sharing an in-memory sequence across multiple worker processes without
    duplicating memory::

      import grain

      # Store the sequence in OS shared memory. Unlike a standard Python list
      # that is copied into every worker process when pickled, only a
      # lightweight shared memory reference is sent to each worker.
      data = [10, 20, 30, 40]
      source = grain.sources.SharedMemoryDataSource(data)

      # Read and transform elements across 2 worker processes.
      ds = (
          grain.MapDataset.source(source)
          .map(lambda x: x * 2)
          .to_iter_dataset()
      )

      print(list(ds))
      # [20, 40, 60, 80]

      # Clean up the shared memory block when done.
      source.close()
      source.unlink()
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

    Raises:
      ValueError: If neither `elements` nor `name` is provided.
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
