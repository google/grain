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
"""This module provides functionality to create index datasets for tf.data.

Index datasets only contain a monotonically increasing index, a dataset_index
(in case multiple datasets are mixed), the epoch number and an integer key
referencing to record in the dataset on disk.

We use the following terms:
- A **dataset** is a set of files on disk.
- A **record** is an entry stored in a file on disk. A record has a position
  inside a specific file and a **record_key** that is unique it's dataset. This
  modules only deals with record keys assuming that components further down
  the tf.data pipeline can convert the record_key to a (filename, position) and
  read the corresponding record.
- An **example** is a transformed record (or a combination of multiple records).
  Input pipelines might iterate over datasets multiple times (also called
  epochs) in which case multilple examples are derived from the same record
  (usually by applying random transformations as data augmentation).
- **Index** is a monotonically increasing integer that indicates how far we have
  advanced our input pipeline (defined using tf.data). Keeping track of the last
  seen index allows users to restart a pipeline at the same position. If we
  shard the dataset across multiple workers the index will still be globally
  unique and each index will map to a spefic worker. For 2 shards (num_shards=2)
  this means worker 1 (shard_id=0) will visit indices [0, 2, 4, ...] and worker
  2 (shard_id=1) will visit indices [1, 3, 5, ...].

An index dataset contains elements that each have an index, a dataset index
(only if mixing multiple datasets) and a record key. We can use the dataset
index and the record key to look up the corresponding example on disk.
Optional the index dataset can also contain the epoch for each element and a
unique random seed (which will change every epoch).

In this module we assume that we know the number of records the datasets and
that records are keyed by an integer in [0, num_records_in_dataset).

**For examples of the output for read the test cases.**
"""
import dataclasses
from typing import Sequence, List, Tuple, Union, Optional, Protocol

from grain._src.core import constants
from grain._src.core import sharding
from grain._src.core.config import config
import grain._src.core.random as grain_random
import jax
import numpy as np
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class FirstIndex:
  pass


@dataclasses.dataclass(frozen=True)
class NextIndex:

  last_seen_index: int


Index = Union[int, FirstIndex, NextIndex]


tf_index_shuffle = tf.random.experimental.index_shuffle
tf_random_fold_in = tf.random.experimental.stateless_fold_in


def _shuffle(index: tf.Tensor, *, seed: tf.Tensor,
             num_records: tf.Tensor) -> tf.Tensor:
  """Shuffles the `index` within the interval [0, num_records - 1].

  Args:
    index: An integer scalar tensor >= 0.
    seed: The seed for shuffling as tensor of shape [2] and dtype
      int32/uint32/int64/uint64.
    num_records: The number of records in an epoch. Indices fall into epochs.
      Epoch n contains the indices [n * num_records, (n + 1) * num_records - 1].
      Each epoch is shuffled differently by alternating the `seed`. fold it into
      the `seed`.

  Returns:
    The value at `index` in a random permutation of [0, num_record - 1]. The
    permutation is fixed by `seed` and `epoch = index // num_records`.
  """
  # We use a different random seed per epoch.
  epoch = index // num_records
  seed = tf_random_fold_in(seed, epoch)
  index %= num_records
  # max_index is inclusive.
  return tf_index_shuffle(
      index, seed=seed, max_index=tf.cast(num_records - 1, tf.int64))


def _interleaved_shuffle(index: tf.Tensor,
                         *,
                         seed: tf.Tensor,
                         num_records: tf.Tensor,
                         block_size: Optional[int] = None,
                         parallel_blocks: Optional[int] = None) -> tf.Tensor:
  """Approximate global shuffle by interleaving random blocks.

  # Motivation
  The goal here is to reduce the number of disk seeks that we have to do when
  reading the shuffled indices from disk and still providing an approximate
  global shuffle. This is useful when seeking for each element is slow or even
  infeasible:
  - Records in the dataset are very small (think <10 KiB).
  - The dataset is very large (think >10M records).
  - Storage backend is very slow (can be the case for hard drives).

  # Algorithm
  1) Divide the dataset with indices in [0, num_records - 1] into blocks of size
     B (`block_size` argument).
  2) Further divide the datasets by combining every C blocks
     (`parallel_blocks` argument) to a partition. Partitions have size
     C*B. (In case of num_records%(C*B) > 0 the last partition is smaller and
     treated separately.)
  3) Compute the partition z=index//(C*B) and the position p=index%(C*B) inside
     the partition. p is always in [0, C*B-1].
  4) Compute p' as the value at position p in a random permutation of
     [0, C*B-1]. p' is a random position inside the partition.
  5) Compute the block q=p'//B + z*C. q the global block index.
  6) Compute q' as the value at position q in a random permutation of
     [0, num_records // B]. q' is a random block in our dataset.
  7) Return q' * B + p % B as the new random index.

  The algorithim above can be implemented for one index at a time using
  `tf.random.experimental.index_shuffle()`. We never materialize the potential
  huge list of blocks or positions within partitions.
  We carefully alter the random seed each epoch and use a different random seed
  for each partition to avoid patterns.

  # Why it works
  The algorithm guarantees that each C*B consecutive indices will belong
  to C blocks of size B. The blocks are randomly distributed within the dataset.
  Grain will batch together reads (see --grain_tf_lookup_batch_size). If we
  batch at least C*B reads together we should often read B consecutive records.
  (One could argue that the batched reading functions as a shuffle buffer.)

  # Last partial partition
  Indices that fall in the last partial partition are simply shuffled within the
  last partition. This is ok since we expect C*B << num_records.
  To read full blocks the consumer of the shuffled indices should batch
  roughly block_size * parallel_blocks reads together.

  Args:
    index: See _shuffle()
    seed: See _shuffle().
    num_records: See _shuffle().
    block_size: Size of a block. Larger blocks result in longer sequential reads
      but worse shuffle performane. Defaults to
      grain.config.tf_interleaved_shuffle_block_size.
    parallel_blocks: Number of blocks to read in parallel and interleave.
      Defaults to grain.config.tf_interleaved_shuffle_parallel_blocks.

  Returns:
    The value at `index` in a permutation of [0, num_record - 1]. The
    permutation approximates a random permutation but depending on
    `block_size` and `parallel_blocks` can contain patterns.
  """
  if block_size is None:
    block_size = config.tf_interleaved_shuffle_block_size
  if parallel_blocks is None:
    parallel_blocks = (config.tf_interleaved_shuffle_parallel_blocks)

  epoch = index // num_records
  seed = tf_random_fold_in(seed, epoch)
  index %= num_records

  # We partition the indices in partitions. The last partition might be
  # incomplete and we handle it separately.
  partition_size = block_size * parallel_blocks
  num_partitions = num_records // partition_size

  if index >= num_partitions * partition_size:
    # Handle last incomplete partition by simply doing global shuffle within it.
    # This should be a tiny fraction of the dataset (block_size and
    # parallel_blocks should be small values compared to max_index).
    offset = num_partitions * partition_size
    num_remaining_indices = num_records - offset
    index -= offset
    index = tf_index_shuffle(
        index, seed=seed, max_index=num_remaining_indices - 1)
    return index + offset

  partition_index = index // partition_size
  partition_seed = tf_random_fold_in(seed, partition_index)

  index_in_partition = index % partition_size
  index_in_partition = tf_index_shuffle(
      index_in_partition, seed=partition_seed, max_index=partition_size - 1)
  index_in_block = index_in_partition % block_size

  num_blocks = num_partitions * parallel_blocks
  block_index = index_in_partition // block_size + partition_index * parallel_blocks
  block_index = tf_index_shuffle(
      block_index, seed=seed, max_index=num_blocks - 1)
  return block_index * block_size + index_in_block


def _float_to_int_proportions(values: Sequence[Union[float, int]],
                              scale_min_to: int = 100) -> List[int]:
  """Scales at values by `scale_min_to/min(proportions)` and cast to int."""
  scale_factor = scale_min_to / min(values)
  return [int(p * scale_factor) for p in values]


def _counts_per_dataset(k: tf.Tensor,
                        proportions: Sequence[int]) -> List[tf.Tensor]:
  """Calculates the counts per dataset at n elements accordings to proportions.

  We are interleaving n infinite datasets into one combined dataset.

  Proportions P is a list of n integers, representing mixing proportions.

  mix(P, k, i) represents the number of examples from component i
  among the first k examples from the mixed sequence. It is given by the
  following formula:

    mix(P, k, 0) = ceiling(k * P[0] / sum(P))
    mix(P, k, i>0) = mix(P[1:], k - mix(P, k, 0), i - 1)

  Element k of the mixed sequence is equal to element m from component i iff:

    mix(P, k + 1, i) == m + 1  AND
    mix(P, k, i) == m

  _counts_per_dataset() computes the "mix" function described above.

  _dataset_and_key_of_next_element() maps from the index in the combined
  dataset to identity of the ID of the source dataset and key in the source
  dataset.

  Args:
    k: Number of elements of the mixed sequence.
    proportions: The mixing proportions for the n dataset.

  Returns:
    Counts of how many elements from each source dataset are used.
  """
  remaining_proportions = sum(proportions)
  result = []
  for p in proportions:
    new_k = (k * (remaining_proportions - p)) // remaining_proportions
    result.append(k - new_k)
    remaining_proportions -= p
    k = new_k
  return result


def _dataset_and_key_of_next_element(
    k: tf.Tensor, proportions: Sequence[int]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Compute the dataset and the key for interleaved datasets at position k.

  We are interleaving n infinite datasets into one combined dataset.

  See the description in _counts_per_dataset() above.

  Args:
    k: Index in the combined dataset.
    proportions: The mixing proportions for the n dataset.

  Returns:
    A tuple with the index of the source dataset and the key in it for the
    element at index `k` of the combined dataset.
  """
  old_counts = tf.stack(_counts_per_dataset(k, proportions))
  new_counts = tf.stack(_counts_per_dataset(k + 1, proportions))
  # For the new dataset the count increased by 1. All other counts should be
  # the same.
  dataset_index = tf.math.argmax(new_counts - old_counts)
  return dataset_index, new_counts[dataset_index] - 1


def _get_shard_size_and_offset(
    num_records: int, options: sharding.ShardOptions) -> tuple[int, int]:
  shard_start, shard_end = sharding.even_split(num_records, options)
  return shard_end - shard_start, shard_start


def _get_shard_sizes_and_offsets_for_mixture(
    records_per_dataset: Sequence[int],
    options: sharding.ShardOptions) -> tuple[Sequence[int], Sequence[int]]:
  # Shard each dataset separately.
  shard_starts, shard_ends = zip(
      *[sharding.even_split(n, options) for n in records_per_dataset])  # pylint: disable=missing-kwoa
  records_per_dataset = [
      end - start for start, end in zip(shard_starts, shard_ends)
  ]
  return records_per_dataset, shard_starts


def _create_index_dataset(records_per_dataset: Union[int, Sequence[int]],
                          *,
                          proportions: Optional[Sequence[Union[int,
                                                               float]]] = None,
                          start_index: Index = FirstIndex(),
                          num_epochs: Optional[int] = None,
                          shuffle: bool = False,
                          seed: Optional[grain_random.RNGKeyLike] = None,
                          shard_options: sharding.ShardOptions,
                          emit_epoch: bool = False,
                          emit_seed: bool = False) -> tf.data.Dataset:
  """Creates a new index dataset.

  See the module description for an explanation of the idea.

  Warning: The current implementation is optimized for speed and only allows
  the start_index to change during the training. If you need to change other
  values during the training and still maintain visitation guarantees please
  contact mrit@.
  (emit_epoch and emit_seed are always safe to change if they extra outputs are
  not used by your preprocessing.)

  The random seed is used both for shuffling and providing a unique random seed
  to each element (see `emit_seed` argument). Below we document the steps
  for deriving random seeds for transparency. Users should not rely on these
  actual steps or their order. However users can rely on the resulting
  guarantees:
  1) If we have a mixture split the seed into a seed for each dataset in the
     mixture.
     Guarantee: Random seeds are independent between datasets in the mixture.
  2) Split the seed(s) into separate seeds for shuffling and preprocessing.
     Guarantee: We always perform this split independent of the `shuffle` and
     `emit_seed` arguments. This means that the settings don't affect each other
     - allowing users to turn on/off shuffling but still get the same
     constants.SEED
     value for each element (just element order is changing).
  3) If there are multiple shards (shard_count>1) fold the shard_id into each
     shuffle seed.
     Guarantee: Each shard should shuffles their part of the data
     in a different order.
  4) When shuffling elements we fold the epoch of the dataset into the shuffle
     seed.
     Guarantee: Each epoch gets a different order.
  5) The constants.SEED feature contains the preprocessing seed (of the dataset)
     folded with the record_key and the epoch.
     Guarantee: Each element gets a unique random seed that changes every epoch.

  Args:
    records_per_dataset: Number of examples per dataset. Provide a sequence to
      mix multiple datasets. If a sequence is provided the output will contain
      the DATASET_INDEX field and the RECORD_KEY points to the record within
      the dataset.
    proportions: Proportions when mixing multiple datasets. If not provided all
      datasets will be mixed with equal proportions. Proportions are relative to
      the sum of all proportions each other at both float and integers can be
      mixed. E.g. when mixing two datasets both [0.25, 0.75] and [1, 3] result
      in the same ratio of 1:3 between the first and the second dataset.
    start_index: Where to start the input pipeline. This can be an integer,
      FirstIndex() or NextIndex(last_seen_index). The latter 2 to are handy for
      distributed settings that shard the data between processes
      (shard_count > 1).
      If an integer is provided it must be >= 0 and be dividable by
      `shard_count`.
    num_epochs: Integer if iterating over a fixed number of epochs. The dataset
      will be finite and have known size. Not supported when mixing multiple
      datasets.
    shuffle: Whether to shuffle record keys. If True you need to provide `seed`.
    seed: Random seed to use for shuffling. This should be a tensor of shape
      [2].
    shard_options: Options for sharding the data. Use `grain.NoSharding()`
      if you don't want to shard the data.
    emit_epoch: If True emits an additional feature EPOCH that counts the
      number of times a record_key (within a dataset) has been visited. This
      starts with 1 and then increases every epoch (there is no epoch 0). When
      mixing multiple datasets each datasets has its own epochs.
    emit_seed: If True emits an additional feature SEED that can be passed to
      stateless random operations (e.g. `tf.random.stateless_uniform()`). The
      seed only depends on the `seed` passed to this function, the RECORD_KEY
      and the EPOCH of the record. It is not affected by shuffling.
      If users need multiple random seeds they can split the SEED using
      `tf.random.experimental.stateless_split()`.

  Returns:
    A `tf.data.Dataset` containing Dict[str, tf.Tensor]. The dictionary will
    contain an INDEX, RECORD_KEY. When mixing multiple datasets it will
    also contain the DATASET_INDEX.
  """
  # Need seed if shuffling.
  if shuffle and seed is None:
    raise ValueError("Shuffling requires specifying a seed.")
  if emit_seed and seed is None:
    raise ValueError("Emitting seed requires specifying a global seed.")

  is_mixture = not isinstance(records_per_dataset, int)
  shard_index = shard_options.shard_index
  shard_count = shard_options.shard_count

  if shuffle or emit_seed:
    # Convert to valid RNGKey.
    assert seed is not None  # See if statements above.
    seed = grain_random.as_rng_key(seed)
    # Split seed for each dataset.
    if is_mixture:
      num_datasets = len(records_per_dataset)
      seed = jax.random.split(seed, num_datasets)
    else:
      seed = [seed]
    # Split into shuffle and preprocess seeds.
    shuffle_seed, preprocess_seed = zip(*[jax.random.split(x) for x in seed])
    if shard_count > 1:
      shuffle_seed = [jax.random.fold_in(x, shard_index) for x in shuffle_seed]
    # Convert to TF random seeds.
    shuffle_seed = tf.constant(np.asarray(shuffle_seed, np.int32))
    preprocess_seed = tf.constant(np.asarray(preprocess_seed, np.int32))
  del seed

  # When sharding the dataset (`shard_count>1`) indices are global and each
  # process will start with its `shard_index` and step by `shard_count`.
  # To make it easier for users to specify a valid `start_index` we allow to
  # special values (and convert them to their integer value below):
  # - FirstIndex() always points to the very first index for the shard.
  # - NextIndex() contains the previus index and will start from the following
  #   index.
  if isinstance(start_index, FirstIndex):
    start_index = shard_index
  elif isinstance(start_index, NextIndex):
    start_index = start_index.last_seen_index + shard_count
  if start_index < 0 or start_index % shard_count != shard_index:
    raise ValueError(
        f"Start index {start_index} is not valid index for {shard_options=} "
        f"start_index % shard_count should equal the shard_index.")

  if num_epochs is None:
    end_index = np.iinfo(np.int64).max
  else:
    if is_mixture:
      # What is an epoch when mixing multiple datasetsw with different number
      # of records or proportions?
      raise ValueError(
          "Using fixed number of epochs is not allowed when mixing datasets.")
    assert isinstance(records_per_dataset, int)
    assert num_epochs is not None
    end_index = records_per_dataset * num_epochs

  # Preparations for sharding.
  if shard_count > 1:
    if is_mixture:
      shard_fn = _get_shard_sizes_and_offsets_for_mixture
    else:
      shard_fn = _get_shard_size_and_offset
    records_per_dataset, position_offset_per_dataset = shard_fn(
        records_per_dataset, options=shard_options)

  # Preparations for mixing by turning several parameters into vectors.
  if is_mixture:
    num_datasets = len(records_per_dataset)  # pytype: disable=wrong-arg-types
    if proportions is None:
      proportions = num_datasets * [1]
    else:
      assert len(proportions) == num_datasets
      proportions = _float_to_int_proportions(proportions)

  shuffle_fn = (
      _interleaved_shuffle if config.tf_interleaved_shuffle else _shuffle)

  # We define one map function that goes from index to global index, position
  # and dataset_index.
  # Note: Please use tf.int64 everywhere to avoid type mismatch errors.
  if is_mixture:

    # Turn lists into tensors. This way we can do lookup using slicing.
    records_per_dataset = tf.stack(records_per_dataset)
    if shard_count > 1:
      position_offset_per_dataset = tf.stack(position_offset_per_dataset)

    def map_fn(index):
      assert index.dtype == tf.int64
      # For shard_count>1 indices are not consecutive. But consecutive numbers
      # required for mixing datasets and performing shuffling.
      local_index = index // shard_count

      dataset_index, index_in_dataset = _dataset_and_key_of_next_element(
          local_index, proportions)
      num_records_in_dataset = tf.cast(records_per_dataset[dataset_index],
                                       tf.int64)
      if shuffle:
        record_key = shuffle_fn(
            index_in_dataset,
            seed=shuffle_seed[dataset_index],
            num_records=num_records_in_dataset)
      else:
        record_key = index_in_dataset % num_records_in_dataset
      if shard_count > 1:
        # Make index global.
        record_key += tf.cast(position_offset_per_dataset[dataset_index],
                              tf.int64)
      metadata = {
          constants.DATASET_INDEX: dataset_index,
          constants.INDEX: index,
          constants.RECORD_KEY: record_key,
      }

      # Add optional elements.
      epoch = index_in_dataset // num_records_in_dataset + 1
      if emit_epoch:
        metadata[constants.EPOCH] = epoch
      if emit_seed:
        metadata[constants.SEED] = tf.random.experimental.stateless_fold_in(
            preprocess_seed[dataset_index], record_key * 2**20 + epoch)

      return metadata
  else:

    def map_fn(index):
      assert index.dtype == tf.int64
      # For shard_count>1 indices are not consecutive. But consecutive numbers
      # required for mixing datasets and performing shuffling.
      local_index = index // shard_count

      if shuffle:
        record_key = shuffle_fn(
            local_index, seed=shuffle_seed[0], num_records=records_per_dataset)
      else:
        record_key = local_index % records_per_dataset
      if shard_count > 1:
        # Make index global.
        record_key += position_offset_per_dataset

      metadata = {constants.INDEX: index, constants.RECORD_KEY: record_key}

      # Add optional elements.
      epoch = local_index // records_per_dataset + 1
      if emit_epoch:
        metadata[constants.EPOCH] = epoch
      if emit_seed:
        metadata[constants.SEED] = tf.random.experimental.stateless_fold_in(
            preprocess_seed[0], record_key * 2**20 + epoch)

      return metadata

  ds = tf.data.Dataset.range(start_index, end_index, shard_count)
  ds = ds.map(map_fn)
  return ds


class TfIndexSampler(Protocol):

  def get_index_dataset(self, start_index: Index) -> tf.data.Dataset:
    """Returns the index dataset starting at start_index."""


@dataclasses.dataclass(frozen=True)
class TfDefaultIndexSampler:
  """Simple index dataset that supports shuffling and sharding.

  See _create_index_dataset().
  """

  num_records: int
  shard_options: sharding.ShardOptions
  shuffle: bool = False
  num_epochs: Optional[int] = None
  seed: Optional[grain_random.RNGKeyLike] = None

  def get_index_dataset(self, start_index: Index) -> tf.data.Dataset:
    return _create_index_dataset(
        self.num_records,
        start_index=start_index,
        num_epochs=self.num_epochs,
        shuffle=self.shuffle,
        seed=self.seed,
        shard_options=self.shard_options,
        emit_epoch=True,
        emit_seed=self.seed is not None)


@dataclasses.dataclass(frozen=True)
class TfMixtureIndexSampler:
  """Simple index dataset that supports shuffling, sharding and mixing.

  See _create_index_dataset().
  """

  records_per_dataset: Sequence[int]
  shard_options: sharding.ShardOptions
  proportions: Optional[Sequence[Union[int, float]]] = None
  shuffle: bool = False
  seed: Optional[grain_random.RNGKeyLike] = None

  def get_index_dataset(self, start_index: Index) -> tf.data.Dataset:
    return _create_index_dataset(
        self.records_per_dataset,
        proportions=self.proportions,
        start_index=start_index,
        shuffle=self.shuffle,
        seed=self.seed,
        shard_options=self.shard_options,
        emit_epoch=True,
        emit_seed=self.seed is not None)
