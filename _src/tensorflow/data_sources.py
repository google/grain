# Copyright 2022 Google LLC
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
"""Data sources provide read access to a dataset.

Data sources handle reading from storage backends like files, databases and
in-memory structures.
Grain provides a few common datasets but users can imlement their own data
sources by implementing the `TfRandomAccessDataSource` protocol.

We do require data sources to have record keys in [0, len(source]). The order
in which records are loaded is decided by the sampler.
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union

from etils import epath
from grain._src.core import usage_logging
from grain._src.tensorflow.ops import array_record_data_source
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

TfArrayRecordDataSource = array_record_data_source.TfArrayRecordDataSource

# TensorFlow function to parse a single record.
TfParseFn = Callable[[tf.Tensor], Mapping[str, Any]]

# For InMemmoryDataSource allow at most 100M elements. Anything above that is
# very likely too big to fit in memory.
_MAX_ROWS_IN_MEMORY = 100_000_000


class TfDataSource(Protocol):
  """Protocol for random access data sources in TensorFlow."""

  def __len__(self) -> int:
    """Number of items in the dataset."""

  def __getitem__(self, record_keys: tf.Tensor) -> Any:
    """Fetch the items with the given record keys using TF ops.

    This function will be compiled using @tf.function.

    Args:
      record_keys: int64 vector with the record_keys to load. Record keys will
        be in [0, len(self)). The DataLoader decides how many records to load in
        parallel. The IndexSampler decides the order of record keys.

    Returns:
      A Tensor or a nested structure of tensors with values for the record keys.
      in the same order. The first dimension of each tensor must have the same
      size is the record_keys.
    """

  def get_parse_fn(self) -> Optional[TfParseFn]:
    """Optional.

    Returns a function to parse a single record.

    This is useful if __getitem__() returns serialized data
    and allows DataLoaders to delay parsing the data (e.g. after mixing).
    """


# TfParseFn for TFDS datasets.
@dataclasses.dataclass(frozen=True)
class _ParseAndDecodeExample:
  """Parse a serialized TF example proto from TFDS."""

  tfds_features: tfds.core.features.FeaturesDict
  decoders: Optional[Mapping[str, Any]] = None

  def __call__(self, record: tf.Tensor) -> Mapping[str, Any]:
    return self.tfds_features.deserialize_example(
        record, decoders=self.decoders)


class TfdsDataSource:
  """Data source for TFDS datasets."""

  def __init__(self,
               dataset_info: tfds.core.DatasetInfo,
               *,
               split: str,
               decoders: Optional[Mapping[str, Any]] = None,
               cache: bool = False):
    self._tfds_info = dataset_info
    self._split = split
    self._decoders = decoders
    file_instructions = dataset_info.splits[split].file_instructions
    paths = [
        f"{fi.filename}[{fi.skip}:{fi.skip + fi.num_examples}]"
        for fi in file_instructions
    ]
    file_format = dataset_info.file_format
    if file_format == tfds.core.file_adapters.FileFormat.ARRAY_RECORD:
      self._source = TfArrayRecordDataSource(paths, cache=cache)
    else:
      raise NotImplementedError("No random access data source for file format "
                                f"{dataset_info.file_format}.")
    usage_logging.log_event(
        "TfdsDataSource", tag_2=file_format.name, tag_3="TfGrain")

  @classmethod
  def from_name(cls,
                name: str,
                *,
                data_dir: Optional[epath.PathLike] = None,
                **kwargs):
    dataset_info = tfds.builder(name, data_dir=data_dir).info
    return cls(dataset_info, **kwargs)

  @classmethod
  def from_directory(cls, directory: epath.PathLike, **kwargs):
    dataset_info = tfds.builder_from_directory(directory).info
    return cls(dataset_info, **kwargs)

  def __len__(self) -> int:
    # We can get this very quickly from TFDS.
    return self._tfds_info.splits[self._split].num_examples

  def __getitem__(self, record_keys: tf.Tensor) -> tf.Tensor:
    return self._source[record_keys]

  def get_parse_fn(self) -> TfParseFn:
    # Turning decoders into a dctionary because TFDS currently doesn't allow
    # immutabledicts
    decoders = dict(self._decoders) if self._decoders else None
    return _ParseAndDecodeExample(self._tfds_info.features, decoders=decoders)

  def __repr__(self) -> str:
    decoders = self._decoders
    if decoders:
      decoders = jax.tree_map(type, decoders)
    return (f"TfdsDataSource(builder_directory={self._tfds_info.data_dir!r}, "
            f"split={self._split!r}, "
            f"decoders={decoders})")


@dataclasses.dataclass
class TfInMemoryDataSource:
  """Data sources that holds all records as tf.Tensor in memory.

  This is useful for small datasets, say less than 10GB. Use one of the
  from_* methods to construct a dataset from your source.
  """

  values: Any  # TF structure (all leaves are tf.Tensor's)
  parse_fn: Optional[TfParseFn] = None

  def __post_init__(self):
    usage_logging.log_event(
        "TfInMemoryDataSource", tag_2="IN_MEMORY", tag_3="TfGrain")

  def __len__(self) -> int:
    return tf.nest.flatten(self.values)[0].shape[0]

  def __getitem__(self, record_keys: tf.Tensor) -> Any:
    record_keys = tf.convert_to_tensor(record_keys)
    if record_keys.shape.rank == 0:
      return tf.nest.map_structure(lambda x: x[record_keys], self.values)
    if record_keys.shape.rank == 1:
      return tf.map_fn(
          lambda r: tf.nest.map_structure(lambda x: x[r], self.values),
          record_keys,
          dtype=tf.nest.map_structure(lambda x: x.dtype, self.values))
    raise ValueError(
        f"Record keys must be a scalar or vector but got {record_keys.rank}.")

  def __repr__(self) -> str:
    element_spec = tf.nest.map_structure(
        lambda x: f"{repr(x.dtype)}{list(x.shape)}", self.values)
    return f"TfInMemoryDataSource({element_spec=}, len={len(self)})"

  def get_parse_fn(self):
    return self.parse_fn

  @classmethod
  def from_dataset(
      cls,
      dataset: tf.data.Dataset,
      *,
      parse_fn: Optional[TfParseFn] = None) -> TfInMemoryDataSource:
    """Constructs an in memory data source from a tf.data.Dataset.

    Args:
      dataset: A dataset with all the elements. This must be finite and elements
        must have the same shape. The dataset will be batched to accelerate
        loading it into memory.
      parse_fn: An optional parsing function to turn serialized example into
        tensors.

    Returns:
      A TfInMemoryDataSource with exactly the same elements in the same order.
    """
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
      raise ValueError("Cannot construct copy an infinite dataset into memory.")
    if dataset.cardinality() > _MAX_ROWS_IN_MEMORY:
      raise ValueError(
          f"Only datasets with at most {_MAX_ROWS_IN_MEMORY!r} elements are "
          f"support but got dataset with cardinality {dataset.cardinality().numpy()}."
      )
    try:
      values = next(iter(dataset.batch(_MAX_ROWS_IN_MEMORY)))
    except Exception as e:
      raise ValueError(
          f"Failed to load {dataset} into memory. Please ensure that the "
          "dataset is valid and all elements have the same shape.") from e
    return cls(values=values, parse_fn=parse_fn)

  @classmethod
  def from_tfds(
      cls,
      name: Optional[str] = None,
      *,
      split: str,
      data_dir: Optional[epath.PathLike] = None,
      tfds_info: Optional[tfds.core.DatasetInfo] = None,
      decoders: Optional[Mapping[str, Any]] = None) -> TfInMemoryDataSource:
    """Constructs an in memory data source from a TFDS dataset."""
    # We cannot use tfds.load() here because we need to separate the raw dataset
    # from the from the parsing.
    if name is None != tfds_info is None:
      raise ValueError("Pass either 'name' or 'tfds_info'.")
    if tfds_info is None:
      tfds_info = tfds.builder(name, data_dir=data_dir).info
    file_instructions = tfds_info.splits[split].file_instructions
    ds = tfds.core.reader._read_files(  # pylint: disable=protected-access
        file_instructions=file_instructions,
        read_config=tfds.ReadConfig(),
        shuffle_files=False,
        disable_shuffling=True,
        file_format=tfds_info.file_format)
    decoders = dict(decoders) if decoders else None
    parse_fn = _ParseAndDecodeExample(tfds_info.features, decoders=decoders)
    return cls.from_dataset(ds, parse_fn=parse_fn)

  @classmethod
  def from_files(cls,
                 filenames: Union[epath.PathLike, Sequence[epath.PathLike]],
                 *,
                 reader_cls: Callable[[tf.Tensor], tf.data.Dataset],
                 parse_fn: Optional[TfParseFn] = None) -> TfInMemoryDataSource:
    """Constructs an in memory data source from a list of files.

    Args:
      filenames: Single filename or list of filenames. The data will be read
        into memory in sequential order (first all records from the first file,
        then all records from the second files, ...).
      reader_cls: Callable for creating a tf.data.Dataset from a single
        filename. Example: tf.data.TFRecordDataset.
      parse_fn: Optional function for parsing a single record.

    Returns:
      A new InMemoryDataSource with the content of the files.
    """
    if isinstance(filenames, (str, pathlib.Path)):
      filenames = [os.fspath(filenames)]
    else:
      filenames = [os.fspath(fn) for fn in filenames]
    ds = tf.data.experimental.from_list(filenames)
    ds = ds.flat_map(reader_cls)
    return cls.from_dataset(ds, parse_fn=parse_fn)

  @classmethod
  def from_data_frame(cls, data_frame) -> TfInMemoryDataSource:
    """Constructs an in memory data source from a Pandas DataFrame."""
    if len(data_frame) > _MAX_ROWS_IN_MEMORY:
      raise ValueError(f"Only DataFrames with at most {_MAX_ROWS_IN_MEMORY!r} "
                       f"rows are support but got {len(data_frame)} rows.")
    ds = tf.data.Dataset.from_tensor_slices(dict(data_frame))
    return cls.from_dataset(ds)
