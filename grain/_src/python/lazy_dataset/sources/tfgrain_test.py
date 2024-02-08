# Copyright 2024 Google LLC
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
"""Tests for TfGrainLazyIterDataset."""

import dataclasses
from typing import Any, Optional

from grain._src.python.lazy_dataset.sources import tfgrain as tfgrain_lazy_dataset
import grain.tensorflow as tfgrain
import tensorflow as tf

from absl.testing import absltest


# Dummy classes copied from data_iterators_test.
# TODO(aaudibert): Refactor into a TfGrain testing.py lib?
@dataclasses.dataclass(frozen=True)
class DummySampler:
  """Dummy IndexSampler that samples range(0, 10)."""

  shard_index: int = 0
  shard_count: int = 1

  def as_dict(self):
    return dataclasses.asdict(self)

  def get_index_dataset(self, start_index: tfgrain.Index) -> tf.data.Dataset:
    if isinstance(start_index, tfgrain.FirstIndex):
      start_index = self.shard_index
    elif isinstance(start_index, tfgrain.NextIndex):
      start_index = start_index.last_seen_index + self.shard_count
    assert isinstance(start_index, int)
    return tf.data.Dataset.range(start_index, 10)


@dataclasses.dataclass(frozen=True)
class DummyDataLoader:
  source: Any = "DummySource"
  sampler: tfgrain.TfIndexSampler = DummySampler()
  batch_size: Optional[int] = None

  def __iter__(self) -> tfgrain.TfGrainDatasetIterator:
    return tfgrain.TfGrainDatasetIterator(self)

  def as_dataset(self, start_index: tfgrain.Index) -> tf.data.Dataset:
    ds = self.sampler.get_index_dataset(start_index)
    ds = ds.map(
        lambda x: {tfgrain.INDEX: x, "number": tf.cast(2 * x + 1, tf.uint32)}
    )
    if self.batch_size:
      ds = ds.batch(self.batch_size, drop_remainder=True)
    return ds


class TfgrainTest(absltest.TestCase):

  def test_basic(self):
    tfgrain_dataset = DummyDataLoader()
    lazy_dataset = tfgrain_lazy_dataset.TfGrainLazyIterDataset(tfgrain_dataset)
    self.assertEqual(list(tfgrain_dataset), list(lazy_dataset))

  def test_checkpoint(self):
    tfgrain_dataset = DummyDataLoader()
    lazy_dataset = tfgrain_lazy_dataset.TfGrainLazyIterDataset(tfgrain_dataset)

    normal_output = list(lazy_dataset)

    interrupted_output = []
    it = iter(lazy_dataset)
    for _ in range(len(normal_output)):
      interrupted_output.append(next(it))
      state = it.get_state()  # pytype: disable=attribute-error

      for _ in range(2):
        try:
          next(it)
        except StopIteration:
          break
      it.set_state(state)  # pytype: disable=attribute-error

    self.assertEqual(normal_output, interrupted_output)


if __name__ == "__main__":
  absltest.main()
