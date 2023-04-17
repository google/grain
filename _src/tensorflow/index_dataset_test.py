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
"""Unit tests for the index_dataset module."""
import collections
import contextlib
import functools
import itertools
from typing import List, Sequence, Union

from absl.testing import parameterized
from grain._src.core.constants import INDEX, RECORD_KEY, DATASET_INDEX, EPOCH, SEED  # pylint: disable=g-multiple-import
from grain._src.core.sharding import ShardOptions
from grain._src.tensorflow import index_dataset
import jax
import numpy as np
import tensorflow as tf

# Some options for testing determinism and restart behavior.
# For each setting there are 2 options and then we test all combinations.
_RECORDS_PER_DATASET = (7, [7, 5])
_PROPORTIONS = (None, [0.3, 0.4])
_SHUFFLE = (True, False)
_SHARD_COUNT = (1, 3)


def create_dataset(
    records_per_dataset: Union[int, Sequence[int]],
    /,
    *,
    emit_epoch: bool = True,
    seed=32,
    shard_before_shuffle: bool = True,
    **kwargs,
):
  """Shortcut for create_index_dataset() that sets emit_epoch and seed."""
  if "shard_options" not in kwargs:
    kwargs["shard_options"] = ShardOptions(0, 1)
  if shard_before_shuffle:
    return index_dataset._create_index_dataset(
        records_per_dataset, emit_epoch=emit_epoch, seed=seed, **kwargs
    )
  return index_dataset._create_index_dataset_v2(
      records_per_dataset, emit_epoch=emit_epoch, seed=seed, **kwargs
  )


class StartIndexTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for StartIndex subclasses."""

  def test_first_index(self):
    rng = np.random.default_rng()
    for _ in range(20):
      shard_count = rng.integers(1, 4, size=1)[0]
      shard_index = rng.integers(shard_count, size=1)[0]
      start_index = index_dataset.FirstIndex()
      start_index = start_index.get_start_index(shard_index, shard_count)
      self.assertEqual(start_index, shard_index)

  def test_next_index(self):
    rng = np.random.default_rng()
    for _ in range(20):
      shard_count = rng.integers(1, 4, size=1)[0]
      shard_index = rng.integers(shard_count, size=1)[0]
      last_seen_index = rng.integers(100, size=1)[0]
      start_index = index_dataset.NextIndex(last_seen_index)
      if last_seen_index % shard_count == shard_index:
        # last_seen_index is valid!
        start_index = start_index.get_start_index(shard_index, shard_count)
        self.assertEqual(start_index, last_seen_index + shard_count)
        self.assertEqual(start_index % shard_count, shard_index)
      else:
        with self.assertRaises(ValueError):
          start_index = start_index.get_start_index(shard_index, shard_count)

  def test_next_valid_index(self):
    rng = np.random.default_rng()
    for _ in range(20):
      shard_count = rng.integers(1, 4, size=1)[0]
      shard_index = rng.integers(shard_count, size=1)[0]
      last_seen_index = rng.integers(100, size=1)[0]
      start_index = index_dataset.NextValidIndex(last_seen_index)
      start_index = start_index.get_start_index(shard_index, shard_count)
      self.assertGreater(start_index, last_seen_index)
      self.assertEqual(start_index % shard_count, shard_index)


class IndexDatasetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for create_index_dataset method."""

  def test_simple(self):
    """No shuffling, no sharding, no mixing."""
    dataset = create_dataset(6)
    values = list(dataset.take(15).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 5},
                         # Second epoch.
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 0},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 1},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 3},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 4},
                         {INDEX: 11, EPOCH: 2, RECORD_KEY: 5},
                         # Third epoch.
                         {INDEX: 12, EPOCH: 3, RECORD_KEY: 0},
                         {INDEX: 13, EPOCH: 3, RECORD_KEY: 1},
                         {INDEX: 14, EPOCH: 3, RECORD_KEY: 2}])
    # pyformat: enable

  def test_emit_seed(self):
    dataset = create_dataset(5, emit_seed=False)
    self.assertNotIn(SEED, dataset.element_spec)
    dataset = create_dataset(5, emit_seed=True)
    self.assertIn(SEED, dataset.element_spec)

  def test_emit_epoch(self):
    dataset = create_dataset(5, emit_epoch=False)
    self.assertNotIn(EPOCH, dataset.element_spec)
    dataset = create_dataset(5, emit_epoch=True)
    self.assertIn(EPOCH, dataset.element_spec)

  def test_num_epochs(self):
    """Setting the number of epochs yields a finite dataset."""
    dataset = create_dataset(4, num_epochs=2)
    values = list(dataset.take(15).as_numpy_iterator())
    self.assertEqual(dataset.cardinality(), 8)
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 3},
                         # Second epoch.
                         {INDEX: 4, EPOCH: 2, RECORD_KEY: 0},
                         {INDEX: 5, EPOCH: 2, RECORD_KEY: 1},
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 3}])
    # pyformat: enable

  def test_num_epochs_none(self):
    """Setting the number of epochs yields a finite dataset."""
    dataset = create_dataset(4, num_epochs=None)
    self.assertEqual(dataset.cardinality(), tf.data.INFINITE_CARDINALITY)

  def test_num_epochs_with_mixture_fails(self):
    """Mixing datasets with fixed number epochs is not allowed."""
    with self.assertRaisesRegex(
        ValueError,
        "Using fixed number of epochs is not allowed when mixing datasets.",
    ):
      index_dataset._create_index_dataset(
          [4, 8], num_epochs=2, shard_options=ShardOptions(0, 1)
      )

  def test_shuffle_simple(self):
    """Shuffling, no sharding, no mixing."""
    dataset = create_dataset(5, shuffle=True, seed=32)
    values = list(dataset.take(10).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 0},
                         # Second epoch.
                         {INDEX: 5, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 0},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 1},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 3},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 4}])
    # pyformat: enable

  @parameterized.parameters(
      (681,),
      (5060,),
  )
  def test_shuffle(self, num_records: int):
    # Get record keys for 3 epochs using various random seeds. All should be
    # shuffled and have a unique order (unless we are very unlucky).
    num_epochs = 3
    unshuffled_record_keys = np.arange(num_records)
    seen_orders = set()
    for seed in [68, 69, (35234, 4050)]:
      # Get 3 epochs and check that each is shuffled and unique.
      dataset = create_dataset(num_records, shuffle=True, seed=seed)
      ds_iter = dataset.as_numpy_iterator()
      for _ in range(num_epochs):
        epoch = [next(ds_iter)[RECORD_KEY] for _ in range(num_records)]
        epoch = tuple(epoch)
        self.assertNotAllEqual(unshuffled_record_keys, epoch)
        self.assertNotIn(epoch, seen_orders)
        seen_orders.add(epoch)

  @parameterized.parameters(
      itertools.product(
          [1, 2],
          [
              # No context.
              (None, None),
              # Each separately.
              ("custom", None),
              ("threefry2x32", None),
              ("rbg", None),
              # implementation frist, custom afterwards.
              ("threefry2x32", "custom"),
              ("rbg", "custom"),
              # custom first, implementation afterwards.
              ("custom", "threefry2x32"),
              ("custom", "rbg"),
          ],
      )
  )
  def test_shuffle_other_argument_types(self, shard_count: int, contexts):
    """Test we can pass TF and JAX random keys."""

    def get_ctx(name):
      if name is None:
        return contextlib.nullcontext()
      if name == "threefry2x32":
        return jax.default_prng_impl("threefry2x32")
      if name == "rbg":
        return jax.default_prng_impl("rbg")
      if name == "custom":
        return jax.enable_custom_prng()
      assert False

    with get_ctx(contexts[0]):
      with get_ctx(contexts[1]):
        int_seed = 42
        tuple_seed = (32, 73)
        # Seed is a tensor.
        tf_seed, _ = tf.unstack(
            tf.random.experimental.stateless_split((32, 73))
        )
        # Seed is a JAX PRNGKey.
        jax_seed = jax.random.PRNGKey(32)
        # Users might have custom PRNG enabled. We test both combinations.
        for seed in [int_seed, tuple_seed, tf_seed, jax_seed]:
          dataset = create_dataset(
              6,
              shuffle=True,
              seed=seed,
              shard_options=ShardOptions(0, shard_count),
          )
          values = list(dataset.take(6).as_numpy_iterator())
          if shard_count == 1:
            self.assertCountEqual(
                [r[RECORD_KEY] for r in values], [0, 1, 2, 3, 4, 5]
            )
          else:
            self.assertCountEqual(
                [r[RECORD_KEY] for r in values], [0, 0, 1, 1, 2, 2]
            )

  def test_sharding_drop_remainder(self):
    dataset = create_dataset(
        8, shard_options=ShardOptions(0, 3, drop_remainder=True)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 0},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 1}])
    # pyformat: enable
    dataset = create_dataset(
        8, start_index=1, shard_options=ShardOptions(1, 3, drop_remainder=True)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 1, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 3}])
    # pyformat: enable
    dataset = create_dataset(
        8, start_index=2, shard_options=ShardOptions(2, 3, drop_remainder=True)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 2, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 5},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 4},
                         {INDEX: 11, EPOCH: 2, RECORD_KEY: 5}])
    # pyformat: enable

  def test_sharding_no_drop_remainder(self):
    dataset = create_dataset(
        8, shard_options=ShardOptions(0, 3, drop_remainder=False)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 6, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 0}])
    # pyformat: enable
    dataset = create_dataset(
        8, start_index=1, shard_options=ShardOptions(1, 3, drop_remainder=False)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 1, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 7, EPOCH: 1, RECORD_KEY: 5},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 3}])
    # pyformat: enable
    dataset = create_dataset(
        8, start_index=2, shard_options=ShardOptions(2, 3, drop_remainder=False)
    )
    values = list(dataset.take(4).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 2, EPOCH: 1, RECORD_KEY: 6},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 7},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 6},
                         {INDEX: 11, EPOCH: 2, RECORD_KEY: 7}])
    # pyformat: enable

  def test_mixing_equal_probability(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = create_dataset([4, 6])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch for both datasets.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 0},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 0},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 6, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 0},
                         {INDEX: 7, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 1},
                         # First dataset is finished and starts second epoch.
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 9, EPOCH: 1, RECORD_KEY: 4, DATASET_INDEX: 1},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 0},
                         {INDEX: 11, EPOCH: 1, RECORD_KEY: 5, DATASET_INDEX: 1},
                         # Second dataset also starts second epoch.
                         {INDEX: 12, EPOCH: 2, RECORD_KEY: 2, DATASET_INDEX: 0},
                         {INDEX: 13, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 14, EPOCH: 2, RECORD_KEY: 3, DATASET_INDEX: 0},
                         {INDEX: 15, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 1},
                        ])
    # pyformat: enable

  def test_mixing_with_integer_proportions(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = create_dataset([3, 4], proportions=[1, 4])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch for both datasets.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 0},
                         # Second dataset is finished and starts second epoch.
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 10, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 0},
                         # Second dataset starts third epoch.
                         {INDEX: 11, EPOCH: 3, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 12, EPOCH: 3, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 13, EPOCH: 3, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 14, EPOCH: 3, RECORD_KEY: 3, DATASET_INDEX: 1},
                         # First dataset is finished and starts second epoch.
                         {INDEX: 15, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 0},
                        ])
    # pyformat: enable

  def test_mixing_with_float_proportions(self):
    """Test mixing with 2 datasets with 4 and 6 elements."""
    dataset = create_dataset([3, 4], proportions=[0.2, 0.8])
    values = list(dataset.take(16).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        # First epoch for both datasets.
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 1, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 0},
                         # Second dataset is finished and starts second epoch.
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 10, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 0},
                         # Second dataset starts third epoch.
                         {INDEX: 11, EPOCH: 3, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 12, EPOCH: 3, RECORD_KEY: 1, DATASET_INDEX: 1},
                         {INDEX: 13, EPOCH: 3, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 14, EPOCH: 3, RECORD_KEY: 3, DATASET_INDEX: 1},
                         # First dataset is finished and starts second epoch.
                         {INDEX: 15, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 0},
                        ])
    # pyformat: enable

  def test_shard_before_shuffle(self):
    dataset = create_dataset(
        6, shuffle=True, seed=32, shard_options=ShardOptions(0, 2)
    )
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 0},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 1},
                         # Second epoch.
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 1},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 0}])
    # pyformat: enable
    dataset = create_dataset(
        6,
        shuffle=True,
        seed=32,
        start_index=1,
        shard_options=ShardOptions(1, 2),
    )
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 1, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 5},
                         # Second epoch.
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 3},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 4},
                         {INDEX: 11, EPOCH: 2, RECORD_KEY: 5}])
    # pyformat: enable

  def test_shard_after_shuffle(self):
    dataset = create_dataset(
        6,
        shuffle=True,
        seed=32,
        shard_options=ShardOptions(0, 2),
        shard_before_shuffle=False,
    )
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 4},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 2},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 5},
                         # Second epoch.
                         {INDEX: 6, EPOCH: 2, RECORD_KEY: 2},
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 1},
                         {INDEX: 10, EPOCH: 2, RECORD_KEY: 5}])
    # pyformat: enable
    dataset = create_dataset(
        6,
        shuffle=True,
        seed=32,
        start_index=1,
        shard_options=ShardOptions(1, 2),
        shard_before_shuffle=False,
    )
    values = list(dataset.take(6).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 1, EPOCH: 1, RECORD_KEY: 3},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 1},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 0},
                         # Second epoch.
                         {INDEX: 7, EPOCH: 2, RECORD_KEY: 0},
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 3},
                         {INDEX: 11, EPOCH: 2, RECORD_KEY: 4}])
    # pyformat: enable

  def test_shard_after_shuffle_change_sharding(self):
    # We iterate for exactly 2 epochs, switching from shard_count=2 to
    # shard_count=3 after a few steps. We should still see every element exactly
    # twice.
    def _create_datasets(shard_count: int, start_index_base=0):
      datasets = []
      for shard_index in range(shard_count):
        ds = create_dataset(
            12,
            shuffle=True,
            seed=32,
            start_index=start_index_base + shard_index,
            shard_options=ShardOptions(shard_index, shard_count),
            shard_before_shuffle=False,
            emit_seed=True,
        )
        datasets.append(ds.as_numpy_iterator())
      return datasets

    # We use the order in which we see elements using a single shard as ground
    # truth.
    ds1 = _create_datasets(1)[0]
    # 24 steps is exactly 2 epochs.
    expected_record_keys, expected_seeds = zip(
        *[(e[RECORD_KEY], tuple(e[SEED])) for _, e in zip(range(24), ds1)]
    )

    record_keys = []
    seeds = []

    # Iterate with 2 shards for 3 steps.
    ds1, ds2 = _create_datasets(2)  # pylint: disable=unbalanced-tuple-unpacking
    for _, e1, e2 in zip(range(3), ds1, ds2):
      record_keys.append(e1[RECORD_KEY])
      seeds.append(tuple(e1[SEED]))
      record_keys.append(e2[RECORD_KEY])
      seeds.append(tuple(e2[SEED]))
    # We saw 6 elements so far, our next valid index will be 6 + shard_index.

    # Iterate with 3 shards for the remaining 6 steps to see 2 full epochs.
    ds1, ds2, ds3 = _create_datasets(3, start_index_base=6)  # pylint: disable=unbalanced-tuple-unpacking
    for _, e1, e2, e3 in zip(range(6), ds1, ds2, ds3):
      record_keys.append(e1[RECORD_KEY])
      seeds.append(tuple(e1[SEED]))
      record_keys.append(e2[RECORD_KEY])
      seeds.append(tuple(e2[SEED]))
      record_keys.append(e3[RECORD_KEY])
      seeds.append(tuple(e3[SEED]))

    # All records keys are valid.
    for x in record_keys:
      self.assertBetween(x, 0, 11)
    self.assertLen(record_keys, 24)
    # We saw exactly 2 epochs.
    self.assertLen(set(record_keys), 12)
    self.assertLen(set(record_keys[:12]), 12)
    self.assertLen(set(record_keys[12:]), 12)
    # All random seeds are unique.
    self.assertLen(set(seeds), 24)
    # Order matches what we expect.
    self.assertAllEqual(expected_record_keys, record_keys)
    # Sharding does not change random seeds.
    self.assertAllEqual(expected_seeds, seeds)

  def test_mixing_and_sharding(self):
    dataset = create_dataset([4, 6], shard_options=ShardOptions(0, 2))
    values = list(dataset.take(10).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 0, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 2, EPOCH: 1, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 4, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 0},
                         {INDEX: 6, EPOCH: 1, RECORD_KEY: 1, DATASET_INDEX: 1},
                         # second epoch of first dataset starts.
                         {INDEX: 8, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 10, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 1},
                         {INDEX: 12, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 0},
                         # second epoch of second dataset starts.
                         {INDEX: 14, EPOCH: 2, RECORD_KEY: 0, DATASET_INDEX: 1},
                         {INDEX: 16, EPOCH: 3, RECORD_KEY: 0, DATASET_INDEX: 0},
                         {INDEX: 18, EPOCH: 2, RECORD_KEY: 1, DATASET_INDEX: 1},
                        ])
    # pyformat: enable
    dataset = create_dataset(
        [4, 6], start_index=1, shard_options=ShardOptions(1, 2)
    )
    values = list(dataset.take(10).as_numpy_iterator())
    # pyformat: disable
    self.assertAllEqual(values,
                        [{INDEX: 1, EPOCH: 1, RECORD_KEY: 2, DATASET_INDEX: 0},
                         {INDEX: 3, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 5, EPOCH: 1, RECORD_KEY: 3, DATASET_INDEX: 0},
                         {INDEX: 7, EPOCH: 1, RECORD_KEY: 4, DATASET_INDEX: 1},
                         # second epoch of first dataset starts.
                         {INDEX: 9, EPOCH: 2, RECORD_KEY: 2, DATASET_INDEX: 0},
                         {INDEX: 11, EPOCH: 1, RECORD_KEY: 5, DATASET_INDEX: 1},
                         {INDEX: 13, EPOCH: 2, RECORD_KEY: 3, DATASET_INDEX: 0},
                         # second epoch of second dataset starts.
                         {INDEX: 15, EPOCH: 2, RECORD_KEY: 3, DATASET_INDEX: 1},
                         {INDEX: 17, EPOCH: 3, RECORD_KEY: 2, DATASET_INDEX: 0},
                         {INDEX: 19, EPOCH: 2, RECORD_KEY: 4, DATASET_INDEX: 1},
                        ])
    # pyformat: enable

  @parameterized.parameters(
      (1,),
      (2,),
      (4,),
  )
  def test_different_order_across_datasets_and_shards(self, shard_count: int):
    """Check that each shard and each dataset gets a different order."""
    # We equaly mix 3 datasets with 64 elements each (1 epoch = 192 elements)
    # and iterate 2 epochs. For each dataset index we should see 2 unique
    # sequences and sequences should be different between datasets and shards.
    num_datasets = 3  # We use 3 datasets with 64 records each.
    num_records_per_dataset = num_datasets * [64]
    num_epochs = 2
    num_elements = num_epochs * sum(num_records_per_dataset)

    # We track sequence of record keys within a shard, dataset and epoch.
    # Each new sequence should be unique (differ from all previous seen
    # sequence). This wouldn't hold for very small datasets but 64 elements is
    # large enough.
    seen_orders = set()
    for shard_index in range(shard_count):
      dataset = create_dataset(
          num_records_per_dataset,
          shard_options=ShardOptions(shard_index, shard_count),
          shuffle=True,
          seed=123,
      )
      # 2 epochs.
      dataset = dataset.take(num_elements)
      # Separate record keys orders but dataset index and epoch.
      orders = collections.defaultdict(list)
      for e in dataset.as_numpy_iterator():
        orders[(e[DATASET_INDEX], e[EPOCH])].append(e[RECORD_KEY])

      for record_order in orders.values():
        # Record order contains global keys, but to evaluate the shuffle order
        # of a shard we care about the local order. We can subtract the minimum
        # to get it. If we did everything right we should have a permutation of
        # [0, ..., 64 // shard_count].
        # Otherwise the logic for this test case is wrong.
        record_order = tuple(np.asarray(record_order) - np.min(record_order))
        self.assertAllEqual(
            sorted(record_order), list(range(64 // shard_count))
        )

        # Unless we are very unlucky we should have a new unseen order.
        self.assertNotIn(record_order, seen_orders)
        seen_orders.add(record_order)

  def test_seed_stable(self):
    """Tests that the per example RNG is the same with and without shuffle."""
    n = 6

    def get_record_to_seed_map(**kwargs):
      shard_count = kwargs.get("shard_options", ShardOptions(0, 1)).shard_count
      assert n % shard_count == 0
      ds = create_dataset(n, emit_seed=True, **kwargs)
      ds = ds.take(n // shard_count)
      return {
          r[RECORD_KEY].numpy().item(): r[SEED].numpy().tolist() for r in ds
      }

    # no shuffling, no sharding.
    seeds_1 = get_record_to_seed_map(shuffle=False)

    # shuffling, no sharding.
    seeds_2 = get_record_to_seed_map(shuffle=True)
    self.assertAllEqual(seeds_1, seeds_2)

    # no shuffling, sharding.
    seeds_3 = {}
    for i in range(2):
      seeds_3 |= get_record_to_seed_map(
          shuffle=False,
          start_index=i,
          shard_options=ShardOptions(i, 2, drop_remainder=False),
      )
    self.assertAllEqual(seeds_1, seeds_3)

    # shuffling, sharding.
    seeds_4 = {}
    for i in range(2):
      seeds_4 |= get_record_to_seed_map(
          shuffle=True,
          start_index=i,
          shard_options=ShardOptions(i, 2, drop_remainder=False),
      )
    self.assertAllEqual(seeds_1, seeds_4)

  @parameterized.parameters(
      itertools.product(
          _RECORDS_PER_DATASET, _PROPORTIONS, _SHUFFLE, _SHARD_COUNT
      )
  )
  def test_determinism(
      self, records_per_dataset, proportions, shuffle: bool, shard_count: int
  ):
    """Creating the dataset twice gives the same result."""
    seed = 3 if shuffle else None
    dataset = create_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        shard_options=ShardOptions(0, shard_count),
    )
    values_1 = list(dataset.take(50).as_numpy_iterator())
    dataset = create_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        shard_options=ShardOptions(0, shard_count),
    )
    values_2 = list(dataset.take(50).as_numpy_iterator())
    self.assertAllEqual(values_1, values_2)

  @parameterized.parameters(
      itertools.product(
          _RECORDS_PER_DATASET, _PROPORTIONS, _SHUFFLE, _SHARD_COUNT
      )
  )
  def test_start_index(
      self, records_per_dataset, proportions, shuffle: bool, shard_count: int
  ):
    """We can start anyway and get the same elements."""
    seed = 3 if shuffle else None
    dataset = create_dataset(
        records_per_dataset,
        proportions=proportions,
        shuffle=shuffle,
        seed=seed,
        shard_options=ShardOptions(0, shard_count),
    )
    all_values = list(dataset.take(50).as_numpy_iterator())
    for step in range(1, 30):
      dataset = create_dataset(
          records_per_dataset,
          proportions=proportions,
          shuffle=shuffle,
          seed=seed,
          shard_options=ShardOptions(0, shard_count),
          start_index=step * shard_count,
      )
      values = list(dataset.take(50 - step).as_numpy_iterator())
      self.assertAllEqual(all_values[step:], values)

  @parameterized.parameters([
      (0, 1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}),
      (0, 2, {0, 2, 4, 6, 8}),
      (1, 2, {1, 3, 5, 7, 9}),
      (2, 3, {2, 5, 8}),
  ])
  def test_invalid_start_index(
      self, shard_index: int, shard_count: int, valid_start_indices
  ):
    for start_index in range(10):
      if start_index in valid_start_indices:
        index_dataset._create_index_dataset(
            100,
            start_index=start_index,
            shard_options=ShardOptions(shard_index, shard_count),
        )
      else:
        with self.assertRaises(ValueError, msg=f"start_index={start_index}"):
          index_dataset._create_index_dataset(
              100,
              start_index=start_index,
              shard_options=ShardOptions(shard_index, shard_count),
          )

  @parameterized.parameters([
      (0, 0, 2, [0, 2, 4, 6, 8]),
      (1, 1, 2, [1, 3, 5, 7, 9]),
      (index_dataset.FirstIndex(), 0, 2, [0, 2, 4, 6, 8]),
      (index_dataset.FirstIndex(), 1, 2, [1, 3, 5, 7, 9]),
      (index_dataset.NextIndex(2), 0, 2, [4, 6, 8]),
      (index_dataset.NextIndex(5), 1, 2, [7, 9]),
  ])
  def test_special_start_indices(
      self,
      start_index,
      shard_index: int,
      shard_count: int,
      expected_indices: List[int],
  ):
    ds = index_dataset._create_index_dataset(
        10,
        start_index=start_index,
        shard_options=ShardOptions(shard_index, shard_count),
        num_epochs=1,
    )
    actual_indices = [e[INDEX].numpy().item() for e in ds]
    self.assertAllEqual(actual_indices, expected_indices)

  @parameterized.parameters([
      ((4, 88), 24, 1),
      ((3, 55), 550, 1),
      ((2, 77), 320, 2),
  ])
  def test_shuffle_is_permutation(
      self, seed: tuple[int, int], num_records: int, num_epochs: int
  ):
    ds = tf.data.Dataset.range(num_records * num_epochs)
    shuffle_fn = functools.partial(
        index_dataset._shuffle, seed=seed, num_records=num_records
    )
    ds = ds.map(shuffle_fn, num_parallel_calls=tf.data.AUTOTUNE)
    shuffled_indices = [x.numpy().item() for x in ds]
    self.assertLen(shuffled_indices, num_records * num_epochs)
    self.assertAllGreaterEqual(shuffled_indices, 0)
    self.assertAllLess(shuffled_indices, num_records)
    self.assertLen(set(shuffled_indices), num_records)

  @parameterized.parameters([
      ((4, 88), 24, 1),
      ((3, 55), 550, 1),
      ((2, 77), 320, 2),
  ])
  def test_interleaved_shuffle_is_permutation(
      self, seed: tuple[int, int], num_records: int, num_epochs: int
  ):
    ds = tf.data.Dataset.range(num_records * num_epochs)
    shuffle_fn = functools.partial(
        index_dataset._interleaved_shuffle, seed=seed, num_records=num_records
    )
    ds = ds.map(shuffle_fn, num_parallel_calls=tf.data.AUTOTUNE)
    shuffled_indices = [x.numpy().item() for x in ds]
    self.assertLen(shuffled_indices, num_records * num_epochs)
    self.assertAllGreaterEqual(shuffled_indices, 0)
    self.assertAllLess(shuffled_indices, num_records)
    self.assertLen(set(shuffled_indices), num_records)


if __name__ == "__main__":
  tf.test.main()
