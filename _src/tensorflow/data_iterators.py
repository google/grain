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
"""Data iterators."""
import dataclasses
import hashlib
import json
from typing import Any, Mapping, Optional

from clu.data import dataset_iterator
from etils import epath
from grain._src.core import constants
from grain._src.tensorflow import index_dataset
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

# Dictionary keys used in checkpoints.
_VERSION = "version"
_LAST_SEEN_INDEX = "last_seen_index"
_SOURCE = "source"
_SAMPLER = "sampler"

DatasetIterator = dataset_iterator.DatasetIterator
ArraySpec = dataset_iterator.ArraySpec


@dataclasses.dataclass(frozen=True)
class IteratorOptions:
  """Options for data iterators in this file.

  Attributes:
    drop_grain_meta_features: If True will drop all meta features created by
      Grain. Meta features are grain.INDEX, grain.SEED, grain.EPOCH etc.
    reshape_for_local_devices: If True will reshape all features from shape [B,
      ...] to [D, B // D, ...] where D=jax.local_device_count(). This is useful
      for data parallelism where each process will load data for several JAX
      devices on the same machine.
  """

  drop_grain_meta_features: bool = False
  reshape_for_local_devices: bool = False


def _drop_grain_meta_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
  """Returns the features with any Grain meta features."""
  result = {}
  for k, v in features.items():
    if k not in constants.META_FEATURES:
      result[k] = v
  return result


def _reshape_for_local_devices(element: Any) -> Any:
  """Reshapes features from [B, ...] to [LocalDevices, B//LocalDevices, ...]."""
  device_count = jax.local_device_count()

  def _reshape(x):
    if x.shape[0] is None:
      shape = (device_count, None) + x.shape[1:]
    elif x.shape[0] % device_count != 0:
      raise ValueError(
          f"Cannot reshape {x} for {device_count} local devices. First "
          "dimension must be a multiple of the number of local devices"
      )
    else:
      shape = (device_count, x.shape[0] // device_count) + x.shape[1:]
    if isinstance(x, dataset_iterator.ArraySpec):
      return dataset_iterator.ArraySpec(x.dtype, shape)
    return x.reshape(shape)

  return jax.tree_map(_reshape, element)


def _tensor_spec_to_array_spec(x: tf.TensorSpec) -> dataset_iterator.ArraySpec:
  return dataset_iterator.ArraySpec(
      dtype=x.dtype.as_numpy_dtype, shape=tuple(x.shape)
  )


class TfGrainDatasetIterator(dataset_iterator.DatasetIterator):
  """Checkpointable iterator that restores state based on the last seen index."""

  def __init__(self, data_loader, options: Optional[IteratorOptions] = None):
    self._data_loader = data_loader
    self._options = options or IteratorOptions()
    # We create these only when needed the first time.
    self._dataset = None
    self._start_index = None
    self._iterator = None
    self._last_seen_index = None

  def _ensure_iterator(self):
    """If missing creates the iterator."""
    if self._iterator is None:
      if self._last_seen_index is None:
        start_index = index_dataset.FirstIndex()
      else:
        start_index = index_dataset.NextIndex(self._last_seen_index)
      # Only recreate the dataset if the start index changed. This enables
      # caching using tf.data which can be useful to speed up evaluation:
      # The evaluation would always start from the beginning (start_index=0,
      # calling reset() before each run.). By reusing the tf.data.Dataset object
      # the CacheTransform can keep computed data in memory across runs.
      if start_index != self._start_index:
        self._start_index = start_index
        self._dataset = self._data_loader.as_dataset(start_index=start_index)
      if not isinstance(self._dataset.element_spec, Mapping):
        raise ValueError(
            "IndexBasedDatasetIterator expect dataset elements to be "
            f"dictionaries but got {self._dataset.element_spec}."
        )
      self._iterator = iter(tfds.as_numpy(self._dataset))

  def __next__(self) -> dataset_iterator.Element:
    self._ensure_iterator()
    element = next(self._iterator)  # Might raise StopIteration.
    self._last_seen_index = element[constants.INDEX].max().item()
    # Apply options.
    if self._options.drop_grain_meta_features:
      element = _drop_grain_meta_features(element)
    if self._options.reshape_for_local_devices:
      element = _reshape_for_local_devices(element)
    return element

  def reset(self):
    self._iterator = None
    self._last_seen_index = None

  @property
  def element_spec(self) -> dataset_iterator.ElementSpec:
    self._ensure_iterator()
    element_spec = jax.tree_map(
        _tensor_spec_to_array_spec, self._dataset.element_spec
    )
    # Apply options.
    if self._options.drop_grain_meta_features:
      element_spec = _drop_grain_meta_features(element_spec)
    if self._options.reshape_for_local_devices:
      element_spec = _reshape_for_local_devices(element_spec)
    return element_spec

  def get_state(self) -> bytes:
    return json.dumps(
        {
            _VERSION: 2,
            _LAST_SEEN_INDEX: self._last_seen_index,
            _SOURCE: repr(self._data_loader.source),
            _SAMPLER: self._data_loader.sampler.as_dict(),
        },
        indent=4,
    ).encode()

  def set_state(self, state: bytes):
    self.reset()
    state = json.loads(state.decode())
    self._last_seen_index = state[_LAST_SEEN_INDEX]
    version = state.get(_VERSION, 0)

    # Fix state of TfArrayRecordDataSource to version 1.
    if version == 0 and state[_SOURCE].startswith(
        "TfArrayRecordDataSource(paths=['"
    ):
      # Extract '[...]' from 'TfArrayRecordDataSource(paths=[...])' and replace
      # ' and " to get a valid JSON list.
      paths = state[_SOURCE][30:-1].replace("'", '"')
      paths = json.loads(paths)
      h = hashlib.sha1()
      for p in paths:
        h.update(p.encode())
      state[_SOURCE] = f"TfArrayRecordDataSource(hash_of_paths={h.hexdigest()})"
    if version == 1:
      state[_SOURCE] = repr(self._data_loader.source)

    # Check that checkpoint is valid.
    if repr(self._data_loader.source) != state[_SOURCE]:
      raise ValueError(
          "Source specification in checkpoint doesn't match expected "
          "specification. Restoring checkpoints for different source is not "
          "supported.\n"
          f"Source:               {repr(self._data_loader.source)}\n"
          f"Source in checkpoint: {state[_SOURCE]}"
      )
    if self._data_loader.sampler.as_dict() != state[_SAMPLER]:
      raise ValueError(
          "Sampler specification in checkpoint doesn't match expected "
          "specification. Restoring checkpoints for different samplers is "
          "currently not supported.\n"
          f"Sampler: {self._data_loader.sampler.as_dict()}\n"
          f"Sampler in checkpoint: {state[_SAMPLER]}"
      )

  def save(self, filename: epath.PathLike):
    """Saves the state of this iterator to filename.

    We store the last seen index. This is enough to restore the iterator.

    We do not serialize the data loader (which can contain arbitrary Python
    objects). The checkpoint can only be restored from an equivalent data
    loader. Especially the data source and the sampler must match.
    To catch some user errors we store some info about the source and sampler.

    Args:
      filename: Path to filename. Parent directory must exist. Checkpoints are
        json files.
    """
    filename = epath.Path(filename)
    filename.write_text(self.get_state().decode())

  def restore(self, filename: epath.PathLike):
    filename = epath.Path(filename)
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    self.set_state(filename.read_text().encode())

  def __repr__(self) -> str:
    return (
        f"DataIterator(data_loader={self._data_loader!r}, "
        f"options={self._options!r}, "
        f"last_seen_index={self._last_seen_index!r})"
    )
