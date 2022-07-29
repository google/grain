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
from typing import Callable, Protocol

import grain._src.core.constants as gc
from jax.experimental import multihost_utils
import tensorflow as tf

TfBatchFn = Callable[[tf.data.Dataset], tf.data.Dataset]


class TfNoBatchFn(Protocol):
  """Dummy function to not perform batching."""

  def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds


@dataclasses.dataclass(frozen=True)
class TfDefaultBatchFn:
  """Simple batching operation. See tf.data.Dataset.batch() for details."""

  batch_size: int
  drop_remainder: bool

  def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.batch(self.batch_size, drop_remainder=self.drop_remainder)


@dataclasses.dataclass(frozen=True)
class TfPadDatasetAndBatchFn:
  """Pad dataset to have the same cardinality across all hosts before batching.

  This is mostly useful for evaluation in a multihost environment (like TPU
  pods) where you neitherwant to drop elements nor can evaluate allow partial
  batches.
  The batched dataset will have a `mask_key` feature which is False of padded
  elements and True otherwise.
  """

  batch_size: int
  mask_key: str

  def __call__(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    local_cardinality = ds.cardinality().numpy()
    max_cardinality = multihost_utils.process_allgather(local_cardinality).max()
    # Round up to the next full batch.
    max_cardinality = -(-max_cardinality // self.batch_size) * self.batch_size
    padding = max_cardinality - local_cardinality
    assert padding >= 0, (
        f"Invalid {padding=} for {local_cardinality=} and {max_cardinality=}")

    filler_element = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype)[None], ds.element_spec)
    filler_element[self.mask_key] = [False]
    if gc.INDEX in filler_element:
      filler_element[gc.INDEX] = [-1]
    filler_dataset = tf.data.Dataset.from_tensor_slices(filler_element)
    filler_dataset = filler_dataset.repeat(padding)

    ds = ds.map(
        lambda features: features | {self.mask_key: True},
        num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.concatenate(filler_dataset)
    assert ds.cardinality().numpy() % self.batch_size == 0
    return ds.batch(self.batch_size, drop_remainder=True)
