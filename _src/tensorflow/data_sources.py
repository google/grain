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

import dataclasses
from typing import Any, Callable, Mapping, Optional, Protocol

from etils import epath
from grain._src.core import usage_logging
from grain._src.tensorflow.ops import array_record_data_source
import tensorflow as tf
import tensorflow_datasets as tfds

TfArrayRecordDataSource = array_record_data_source.TfArrayRecordDataSource

# TensorFlow function to parse a single record.
TfParseFn = Callable[[tf.Tensor], Mapping[str, Any]]


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
               decoders: Optional[Mapping[str, Any]] = None):
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
      self._source = TfArrayRecordDataSource(paths)
    else:
      raise NotImplementedError("No random access data source for file format "
                                f"{dataset_info.file_format}.")
    usage_logging.log_event("TfdsDataSource", tag_2=file_format.name)

  @classmethod
  def from_name(cls,
                name: str,
                *,
                split: str,
                data_dir: Optional[epath.PathLike] = None,
                decoders: Optional[Mapping[str, Any]] = None):
    dataset_info = tfds.builder(name, data_dir=data_dir).info
    return cls(dataset_info, split=split, decoders=decoders)

  @classmethod
  def from_directory(cls,
                     directory: epath.PathLike,
                     *,
                     split: str,
                     decoders: Optional[Mapping[str, Any]] = None):
    dataset_info = tfds.builder_from_directory(directory).info
    return cls(dataset_info, split=split, decoders=decoders)

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
    return (f"TfdsDataSource(builder_directory={self._tfds_info.data_dir!r}, "
            f"split={self._split!r}, "
            f"decoders={self._decoders!r})")
