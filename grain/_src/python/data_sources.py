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
import math
from multiprocessing import shared_memory
import os
import threading
import typing
from typing import Any, Generic, Mapping, Optional, Protocol, Sequence, SupportsIndex, TypeVar, Union

from absl import logging
import array_record.python.array_record_data_source as array_record
from etils import epath
from grain._src.core import usage_logging
import tree

# TFDS might not be available if the users did not explicitly depend on it.
try:
  import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

  DatasetInfo = tfds.core.DatasetInfo
except ImportError:
  tfds = None
  DatasetInfo = Any

T = TypeVar("T")
_SLT = TypeVar("_SLT")

_SparseArray = collections.namedtuple(
    "SparseArray", ["indices", "values", "dense_shape"]
)

ArrayRecordDataSource = array_record.ArrayRecordDataSource


@typing.runtime_checkable
class RandomAccessDataSource(Protocol, Generic[T]):
  """Interface for datasources where storage supports efficient random access."""

  def __len__(self) -> int:
    ...

  def __getitem__(self, record_key: SupportsIndex) -> T:
    ...


class RangeDataSource:
  """Range data source, similar to python range() function."""

  def __init__(self, start: int, stop: int, step: int):
    assert step != 0, "step can't be zero."
    self._start = start
    self._stop = stop
    self._step = step
    self._len = int(math.ceil((self._stop - self._start) / step))
    assert self._len >= 0, "length can't be negative."

  def __len__(self) -> int:
    return self._len

  def __getitem__(self, record_key: SupportsIndex) -> int:
    record_key = record_key.__index__()
    assert record_key >= 0 and record_key < self._len
    return self._start + record_key * self._step

  def __repr__(self) -> str:
    return (
        f"RangeDataSource(start={self._start}, stop={self._stop}, "
        f"step={self._step})"
    )


class InMemoryDataSource(shared_memory.ShareableList[_SLT]):
  """Simple in-memory data source for sequences that is sharable among mutiple processes.

  Note:
    This constrains storable values to only the int, float, bool, str (less than
    10M bytes each), bytes (less than 10M bytes each), and None built-in data
    types. It also notably differs from the built-in list type in that these
    lists can not change their overall length (i.e. no append, insert, etc.)
  """

  def __init__(
      self,
      elements: Optional[Sequence[_SLT]] = None,
      *,
      name: Optional[str] = None,
  ):
    """Creates a new InMemoryDataSource object.

    Args:
      elements: the elements for the sharable list.
      name: the name of the datasource.
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


# pytype: disable=attribute-error
class TfdsDataSource:
  """Data source for TFDS datasets.

  # copybara:begin
  Warning: Grain doesn't link in TFDS. If you are using this data source you
  should explicitly depend on //third_party/py/tensorflow_datasets.
  # copybara:end
  """

  def __init__(
      self,
      dataset_info: DatasetInfo,
      *,
      split: str,
      decoders: Optional[Mapping[str, Any]] = None,
  ):
    self._split = split
    self._len = dataset_info.splits[self._split].num_examples
    self._features = dataset_info.features
    self._data_dir = dataset_info.data_dir
    # Turning decoders into a dictionary because TFDS currently doesn't allow
    # `immutabledict`.
    self._decoders = dict(decoders) if decoders else None
    file_format = dataset_info.file_format
    self._file_instructions = dataset_info.splits[split].file_instructions
    if file_format == tfds.core.file_adapters.FileFormat.ARRAY_RECORD:
      self._source = ArrayRecordDataSource(self._file_instructions)
    else:
      raise NotImplementedError(
          "No random access data source for file format "
          f"{dataset_info.file_format}."
      )
    usage_logging.log_event(
        "TfdsDataSource", tag_2=file_format.name, tag_3="PyGrain"
    )

  @classmethod
  def from_name(
      cls, name: str, *, data_dir: Optional[epath.PathLike] = None, **kwargs
  ):
    dataset_info = tfds.builder(name, data_dir=data_dir).info
    return cls(dataset_info, **kwargs)

  @classmethod
  def from_directory(cls, directory: epath.PathLike, **kwargs):
    dataset_info = tfds.builder_from_directory(directory).info
    return cls(dataset_info, **kwargs)

  def __len__(self) -> int:
    return self._len

  def __enter__(self):
    logging.debug("__enter__ for TfdsDataSource is called.")
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    logging.debug("__enter__ for TfdsDataSource is called.")
    self._source.__exit__(exc_type, exc_value, traceback)

  def __getitem__(self, record_key: SupportsIndex) -> int:
    record = self._source[record_key]
    record = self._features.deserialize_example(record, decoders=self._decoders)
    return tree.map_structure(_as_numpy, record)

  def __repr__(self) -> str:
    decoders = self._decoders
    if decoders:
      decoders = tree.map_structure(type, decoders)
    return (
        f"TfdsDataSource(builder_directory={self._data_dir!r}, "
        f"split={self._split!r}, "
        f"decoders={decoders})"
    )

  # pytype: enable=attribute-error

  def __getstate__(self):
    logging.debug("__getstate__ for TfdsDataSource is called.")
    state = self.__dict__.copy()
    del state["_source"]
    return state

  def __setstate__(self, state):
    logging.debug("__setstate__ for TfdsDataSource is called.")
    self.__dict__.update(state)
    self._source = ArrayRecordDataSource(self._file_instructions)
