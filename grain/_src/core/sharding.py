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
"""Classes for handling sharding of data sources arcoss machines/VMs."""
import dataclasses

from absl import logging


@dataclasses.dataclass(frozen=True)
class ShardOptions:
  """Dataclass to hold options for sharding a data source.

  Attributes:
    shard_index: The index of the shard to use in this process. Must be in [0,
      shard_count - 1].
    shard_count: The total number of shards.
    drop_remainder: If True shard() will create even splits and drop the
      remainder examples (all shards will have the same number of examples). If
      False will distribute the remainder N over the first N shards.
  """

  shard_index: int
  shard_count: int
  drop_remainder: bool = False

  def __post_init__(self):
    if self.shard_count <= 0:
      raise ValueError(
          "Number of shards must be a positive integer but got "
          f"{self.shard_count}."
      )
    if self.shard_index < 0 or self.shard_index >= self.shard_count:
      raise ValueError(
          "Shard shard_index must be in [0, shard_count - 1], shard_count was "
          f"{self.shard_count} and shard_index was {self.shard_index}."
      )


class NoSharding(ShardOptions):
  """Doesn't shard data. Each process will load all data."""

  def __init__(self):
    super().__init__(shard_index=0, shard_count=1, drop_remainder=False)


class ShardByJaxProcess(ShardOptions):
  """Shards the data across JAX processes."""

  def __init__(self, drop_remainder: bool = False):
    process_index, process_count = get_process_index_and_count()
    super().__init__(
        shard_index=process_index,
        shard_count=process_count,
        drop_remainder=drop_remainder,
    )


def even_split(num_examples: int, options: ShardOptions) -> tuple[int, int]:
  """Returns the interval for the shard when sharding `num_examples` evenly.

  This splits the interval [0, num_examples - 1] into `shard_count` intervals
  and returns the `shard_index`'s interval. If `drop_remainder` is True all
  intervals will have the same size.

  Args:
    num_examples: Number of examples to shard.
    options: Options for sharding the data in this process.

  Returns:
    Tuple with the start and end of the interval. The start is the first
    example that should be included in this interval and end - 1 is the last
    example to be include in the shard.
  """
  examples_per_shard = num_examples // options.shard_count
  shard_start = examples_per_shard * options.shard_index
  shard_end = examples_per_shard * (options.shard_index + 1)

  # Handle remaining examples.
  num_unused_examples = num_examples % options.shard_count

  if num_unused_examples > 0:
    if options.drop_remainder:
      logging.warning(
          "Dropping %d examples of %d examples (shard %d).",
          num_unused_examples,
          num_examples,
          options.shard_count,
      )
    else:
      shard_start += min(options.shard_index, num_unused_examples)
      shard_end += min(options.shard_index + 1, num_unused_examples)
  return shard_start, shard_end


def get_process_index_and_count():
  try:
    import jax  # pylint:disable=g-import-not-at-top  # pytype:disable=import-error

    return jax.process_index(), jax.process_count()
  except ImportError:
    return 0, 1
