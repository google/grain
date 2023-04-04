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
"""Methods for creating batches of examples.

Batching is usually the last step in an input pipeline. In certain use cases
it can be useful to fuse other operations with batching (e.g padding elements
to the dataset or packing multiple small elemnts together).
"""
import dataclasses
from typing import Any, Mapping, Union

from grain._src.core import constants
from grain._src.tensorflow import transforms
from grain._src.tensorflow.ops import batch_and_pack
from jax.experimental import multihost_utils
import tensorflow as tf


class TfBatchNone(transforms.GlobalTfDataTransform):
  """Dummy function to not perform batching."""

  def apply_to_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds


@dataclasses.dataclass(frozen=True)
class TfBatch(transforms.GlobalTfDataTransform):
  """Simple batching operation. See tf.data.Dataset.batch() for details."""

  batch_size: int
  drop_remainder: bool
  num_parallel_calls: int = tf.data.AUTOTUNE

  def apply_to_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.batch(
        self.batch_size,
        drop_remainder=self.drop_remainder,
        num_parallel_calls=self.num_parallel_calls,
        deterministic=True,
    )


@dataclasses.dataclass(frozen=True)
class TfBatchWithPadElements(transforms.GlobalTfDataTransform):
  """Pad dataset to have the same cardinality across all hosts before batching.

  This is mostly useful for evaluation in a multihost environment (like TPU
  pods) where you neither want to drop elements nor can evaluate on partial
  batches.

  The batched dataset will have a `mask_key` feature which is False of padded
  elements and True otherwise.
  """

  batch_size: int
  mask_key: str
  num_parallel_calls: int = tf.data.AUTOTUNE

  def apply_to_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    local_cardinality = ds.cardinality().numpy()
    if local_cardinality == tf.data.UNKNOWN_CARDINALITY:
      raise ValueError(
          "Dataset has unknown cardinality before batching. This is not "
          "allowed since we statically pad filler elements to get the same "
          "number of batches in all process. Please remove any transformations "
          "that erase the cardinality (e.g. filter)."
      )
    max_cardinality = multihost_utils.process_allgather(local_cardinality).max()
    # Round up to the next full batch.
    max_cardinality = -(-max_cardinality // self.batch_size) * self.batch_size
    padding = max_cardinality - local_cardinality
    assert (
        padding >= 0
    ), f"Invalid {padding=} for {local_cardinality=} and {max_cardinality=}"

    filler_element = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype), ds.element_spec
    )
    filler_element[self.mask_key] = False
    if constants.INDEX in filler_element:
      filler_element[constants.INDEX] = tf.cast(-1, tf.int64)
    filler_dataset = tf.data.Dataset.from_tensors(filler_element)
    filler_dataset = filler_dataset.repeat(padding)

    def mask_fn(features):
      mask_value = tf.convert_to_tensor(features.get(self.mask_key, True))
      if mask_value.dtype != tf.bool or mask_value.shape.rank != 0:
        raise ValueError(
            f"The key {self.mask_key} already exists but its value is not a "
            "boolean scalar."
        )
      return features | {self.mask_key: mask_value}

    ds = ds.map(mask_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.concatenate(filler_dataset)
    assert ds.cardinality().numpy() % self.batch_size == 0
    return ds.batch(
        self.batch_size,
        drop_remainder=True,
        num_parallel_calls=self.num_parallel_calls,
        deterministic=True,
    )


@dataclasses.dataclass(frozen=True)
class TfBatchAndPack(transforms.GlobalTfDataTransform):
  """Fused operation that batches to packed examples.

  See batch_and_pack.py.

  The op has special behavior for the grain.INDEX feature and will always pack
  it, even you it's not included in sequence_lengths. Grains needs the
  grain.INDEX feature to track progress.
  """

  batch_size: int
  sequence_lengths: Any  # Must have the same structure as dataset elements.

  segment_ids_suffix: str = "_segment_ids"
  positions_suffix: str = "_positions"

  def apply_to_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    if not isinstance(ds.element_spec, Mapping):
      raise ValueError(
          "TfBatchAndPack expects elements of the dataset to be dictionaries "
          f"but got {ds.element_spec}."
      )
    for k, v in ds.element_spec.items():
      if not isinstance(v, tf.TensorSpec):
        raise ValueError(
            "TfBatchAndPack expects elements of the dataset to be "
            f"dictionaries containing tensors but got {v} for feature {k}."
        )

    sequence_lengths = dict(self.sequence_lengths)
    if constants.INDEX not in sequence_lengths:
      sequence_lengths[constants.INDEX] = max(tf.nest.flatten(sequence_lengths))

    if constants.INDEX not in ds.element_spec:
      raise ValueError(
          f"The index (key {constants.INDEX}) must still be present for "
          "TfBatchAndPack, else datasets cannot be checkpointed."
      )

    drop_features = set()
    for k in constants.META_FEATURES:
      if k in ds.element_spec and k not in sequence_lengths:
        drop_features.add(k)
    if drop_features:
      ds = ds.map(
          lambda d: {k: v for k, v in d.items() if k not in drop_features},
          num_parallel_calls=tf.data.AUTOTUNE,
      )

    ds = batch_and_pack.BatchAndPackDataset(
        ds, batch_size=self.batch_size, sequence_lengths=sequence_lengths
    )

    # BatchAndPackDataset will replace each feature with 3 features:
    # (values, segment_ids, positions). This is very generic but here we convert
    # back into a flat dictionary.
    def flatten_dict(features):
      result = {}
      for k, (values, segment_ids, positions) in features.items():
        result[k] = values
        # For the INDEX feature we don't need the segmend IDs and positions.
        if k != constants.INDEX:
          result[f"{k}{self.segment_ids_suffix}"] = segment_ids
          result[f"{k}{self.positions_suffix}"] = positions
      return result

    ds = ds.map(flatten_dict, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


TfBatchFn = Union[TfBatchNone, TfBatch, TfBatchWithPadElements, TfBatchAndPack]
