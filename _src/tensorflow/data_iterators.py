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
import json
from typing import Any, Mapping

from clu.data import dataset_iterator
from etils import epath
from grain._src.core import constants
from grain._src.tensorflow import index_dataset
from jax import tree_util
import tensorflow as tf

_LAST_SEEN_INDEX = "last_seen_index"
_SAMPLER_DICT = "sampler_dict"


def _drop_grain_meta_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
  result = {}
  for k, v in features.items():
    if k not in constants.META_FEATURES:
      result[k] = v
  return result


def _tensor_spec_to_array_spec(x: tf.TensorSpec) -> dataset_iterator.ArraySpec:
  return dataset_iterator.ArraySpec(
      dtype=x.dtype.as_numpy_dtype, shape=tuple(x.shape))


class DataIterator(dataset_iterator.DatasetIterator):
  """Checkpointable iterator that restores state based on the last seen index."""

  def __init__(self, data_loader):
    self._data_loader = data_loader
    self._dataset = None
    self._iterator = None
    self._last_seen_index = None

  def _ensure_iterator(self):
    """If missing creates the iterator."""
    if self._iterator is None:
      if self._last_seen_index is None:
        start_index = index_dataset.FirstIndex()
      else:
        start_index = index_dataset.NextIndex(self._last_seen_index)
      self._dataset = self._data_loader.as_dataset(start_index=start_index)
      if not isinstance(self._dataset.element_spec, Mapping):
        raise ValueError(
            "IndexBasedDatasetIterator expect dataset elements to be "
            f"dictionaries but got {self._dataset.element_spec}.")
      self._iterator = self._dataset.as_numpy_iterator()

  def __next__(self) -> dataset_iterator.Element:
    self._ensure_iterator()
    element = next(self._iterator)
    self._last_seen_index = element[constants.INDEX].max().item()
    if self._data_loader.drop_grain_meta_features:
      element = _drop_grain_meta_features(element)
    return element

  def reset(self):
    self._dataset = None
    self._iterator = None
    self._last_seen_index = None

  @property
  def element_spec(self) -> dataset_iterator.ElementSpec:
    self._ensure_iterator()
    element_spec = self._dataset.element_spec
    if self._data_loader.drop_grain_meta_features:
      element_spec = _drop_grain_meta_features(element_spec)
    return tree_util.tree_map(_tensor_spec_to_array_spec, element_spec)

  def save(self, filename: epath.Path):
    state = {
        _LAST_SEEN_INDEX: self._last_seen_index,
        _SAMPLER_DICT: dataclasses.asdict(self._data_loader.sampler)
    }
    state[_SAMPLER_DICT]["__name__"] = type(self._data_loader.sampler).__name__
    filename.write_text(json.dumps(state))

  def restore(self, filename: epath.Path):
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    state = json.loads(filename.read_text())
    expected_sampler_dict = dataclasses.asdict(self._data_loader.sampler)
    expected_sampler_dict["__name__"] = type(self._data_loader.sampler).__name__
    if state[_SAMPLER_DICT] != expected_sampler_dict:
      raise ValueError(
          "Sampler specification in checkpoint doesn't match expected "
          "specification. Restoring checkpoints for different samplers is "
          "currently not supported.\nSampler in checkpoint: "
          f"{state[_SAMPLER_DICT]}\nExpected sampler: {expected_sampler_dict}.")
    self._last_seen_index = state[_LAST_SEEN_INDEX]
