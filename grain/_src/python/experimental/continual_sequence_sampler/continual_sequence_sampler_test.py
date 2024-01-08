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
"""Tests for the ContinualSequenceSampler."""

import bisect
import dataclasses
import itertools
import typing
from typing import Iterable, List, Sequence
from unittest import mock

from absl.testing import absltest
from grain._src.core import sharding
from grain._src.python.experimental.continual_sequence_sampler import continual_sequence_sampler
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import shuffle


@dataclasses.dataclass(frozen=True)
class RecordMetadata:
  index: int
  record_key: int
  element: int
  clip: int


def _get_all_metadata(
    sampler: continual_sequence_sampler.SamplerWrapper,
    gen: Iterable[int],
    uses_bisect: bool,
) -> Sequence[RecordMetadata]:
  """Get metadata for records produced by the sampler for indices from gen."""
  # Mock the bisect_left function so that we can assert that it is called only
  # when we expect it to be. This logarithmic function should only be called if
  # we sample indices out of order.
  bisect_fn = bisect.bisect_left
  with mock.patch.object(bisect, "bisect_left") as mock_bisect:
    mock_bisect.side_effect = bisect_fn
    metadata = []
    for i in gen:
      try:
        record_metadata = sampler[i]
        element_clip = sampler.record_key_to_element_and_clip(
            record_metadata.record_key
        )
        metadata.append(
            RecordMetadata(
                record_metadata.index,
                record_metadata.record_key,
                element_clip.element,
                element_clip.clip,
            )
        )
      except IndexError:
        break
    if uses_bisect:
      mock_bisect.assert_called()
    else:
      mock_bisect.assert_not_called()
  return metadata


class FakeShuffledDataset(lazy_dataset.LazyMapDataset[int]):

  def __init__(self, values: Sequence[int], length: int) -> None:
    self._values = values
    self._length = length

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index):
    return self._values[index]


def index_generator(shard_options: sharding.ShardOptions) -> Iterable[int]:
  return itertools.count(shard_options.shard_index, shard_options.shard_count)


def expected_indices(
    shard_options: sharding.ShardOptions, expected_length: int
) -> List[int]:
  return list(itertools.islice(index_generator(shard_options), expected_length))


class ContinualSequenceSamplerTest(absltest.TestCase):

  def assert_expected_metadata(
      self,
      actual_metadata: Sequence[RecordMetadata],
      expected_index: Sequence[int],
      expected_record_key: Sequence[int],
      expected_element: Sequence[int],
      expected_clip: Sequence[int],
  ) -> None:
    actual_index = [m.index for m in actual_metadata]
    actual_record_key = [m.record_key for m in actual_metadata]
    actual_element = [m.element for m in actual_metadata]
    actual_clip = [m.clip for m in actual_metadata]
    self.assertSequenceEqual(actual_index, expected_index)
    self.assertSequenceEqual(actual_record_key, expected_record_key)
    self.assertSequenceEqual(actual_element, expected_element)
    self.assertSequenceEqual(actual_clip, expected_clip)

  def test_no_videos(self):
    with self.assertRaises(ValueError):
      continual_sequence_sampler.get_sampler(
          clip_map=[],
      )

  def test_fewer_videos_than_shards(self):
    with self.assertRaisesRegex(
        ValueError,
        "Cannot shard with fewer videos than shards. "
        "Num videos = 1, num shards = 2.",
    ):
      continual_sequence_sampler.BatchedContinualSequenceSampler(
          clip_map=[10],
          shard_options=sharding.ShardOptions(shard_index=1, shard_count=2),
      )

  def test_no_shuffling(self):
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=[3, 1, 5],
        num_epochs=2,
    )
    actual_metadata = _get_all_metadata(
        sampler, itertools.count(), uses_bisect=False
    )
    # pyformat: disable
    self.assert_expected_metadata(
        actual_metadata,
        # The index should simply increment matching the indices requested.
        # The total number should be epochs * the number of elements which is
        # the sum of the number of clips in each element (in this case 9).
        expected_index=[0, 1, 2, 3, 4, 5, 6, 7, 8,  # epoch 0
                        9, 10, 11, 12, 13, 14, 15, 16, 17],  # epoch 1
        # The record key should increment but reset each epoch.
        expected_record_key=[0, 1, 2, 3, 4, 5, 6, 7, 8,  # epoch 0
                             0, 1, 2, 3, 4, 5, 6, 7, 8],  # epoch 1
        # The elements should be repeated according to the number of clips in
        # each, i.e. in this case 0 is repeated 3 times, 1 is repeated once and
        # 2 is repeated 5 times.
        expected_element=[0, 0, 0, 1, 2, 2, 2, 2, 2,  # epoch 0
                          0, 0, 0, 1, 2, 2, 2, 2, 2],  # epoch 1
        # The clip number should increment within each element. In this example
        # the first element has 3 clips so we start with 0, 1, 2, before
        # resetting for the next element.
        expected_clip=[0, 1, 2, 0, 0, 1, 2, 3, 4,  # epoch 0
                       0, 1, 2, 0, 0, 1, 2, 3, 4],  # epoch 1
    )
    # pyformat: enable

  def test_out_of_order_indexing(self):
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=[3, 1, 5],
        num_epochs=2,
    )
    # This test is the same as in test_no_shuffling but in this case we access
    # the elements out of order.
    # Just get a couple of elements out of order to ensure that the bisect
    # method is called and works as intended.
    actual_metadata = _get_all_metadata(sampler, [5, 2], uses_bisect=True)
    self.assert_expected_metadata(
        actual_metadata,
        expected_index=[5, 2],
        expected_record_key=[5, 2],
        expected_element=[2, 0],
        expected_clip=[1, 2],
    )

  def test_shuffling(self):
    clip_map = [3, 1, 5]
    # Shuffle the dataset using a fake shuffler where we can specify the order.
    # pyformat: disable
    shuffled_order = [2, 0, 1,  # epoch 0
                      1, 2, 0]  # epoch 1
    # pyformat: enable
    fake_shuffled = FakeShuffledDataset(shuffled_order, len(clip_map))
    fake_shuffled_constructor = lambda *args, **kwargs: fake_shuffled
    with mock.patch.object(
        shuffle,
        "ShuffleLazyMapDataset",
        new_callable=lambda: fake_shuffled_constructor,
    ):
      sampler = continual_sequence_sampler.get_sampler(
          clip_map=clip_map,
          num_epochs=2,
          shuffle_dataset=True,
      )
    actual_metadata = _get_all_metadata(
        sampler, itertools.count(), uses_bisect=False
    )
    # pyformat: disable
    expected_record_key = [4, 5, 6, 7, 8,  # element 2
                           0, 1, 2,  # element 0
                           3,  # element 1
                           3,  # element 1
                           4, 5, 6, 7, 8,  # element 2
                           0, 1, 2]  # element 0
    expected_element = [2, 2, 2, 2, 2,  # element 2
                        0, 0, 0,  # element 0
                        1,  # element 1
                        1,  # element 1
                        2, 2, 2, 2, 2,  # element 2
                        0, 0, 0]  # element 0
    expected_clip = [0, 1, 2, 3, 4,  # element 2
                     0, 1, 2,  # element 0
                     0,  # element 1
                     0,  # element 1
                     0, 1, 2, 3, 4,  # element 2
                     0, 1, 2]  # element 0
    # pyformat: enable
    self.assert_expected_metadata(
        actual_metadata,
        list(range(sum(clip_map) * 2)),
        expected_record_key,
        expected_element,
        expected_clip,
    )

  def test_sharding(self):
    clip_map = [3, 1, 5]
    # Shard 0 gets the largest element
    shard_options = sharding.ShardOptions(shard_index=0, shard_count=2)
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=clip_map,
        shard_options=shard_options,
        num_epochs=2,
    )
    sub_sampler = typing.cast(
        continual_sequence_sampler.BatchedContinualSequenceSampler,
        sampler._sampler,
    )
    self.assertLen(sub_sampler._batch_idx_sampler, 1)
    actual_clip_map = sub_sampler._batch_idx_sampler[0]._clip_map
    self.assertSequenceEqual(actual_clip_map, [5])
    actual_metadata = _get_all_metadata(
        sampler,
        index_generator(shard_options),
        uses_bisect=False,
    )
    # pyformat: disable
    expected_element = [2, 2, 2, 2, 2,  # element 2
                        2, 2, 2, 2, 2]  # element 2
    expected_clip = [0, 1, 2, 3, 4,  # element 2
                     0, 1, 2, 3, 4]  # element 2
    expected_record_key = [4, 5, 6, 7, 8,  # element 2
                           4, 5, 6, 7, 8]  # element 2
    # pyformat: enable
    self.assert_expected_metadata(
        actual_metadata,
        expected_indices(shard_options, len(actual_metadata)),
        expected_record_key,
        expected_element,
        expected_clip,
    )

    # Shard 1 gets the other 2 elements
    shard_options = sharding.ShardOptions(shard_index=1, shard_count=2)
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=clip_map,
        shard_options=shard_options,
        num_epochs=2,
    )
    sub_sampler = typing.cast(
        continual_sequence_sampler.BatchedContinualSequenceSampler,
        sampler._sampler,
    )
    self.assertLen(sub_sampler._batch_idx_sampler, 1)
    actual_clip_map = sub_sampler._batch_idx_sampler[0]._clip_map
    self.assertSequenceEqual(actual_clip_map, [3, 1])
    actual_metadata = _get_all_metadata(
        sampler, index_generator(shard_options), uses_bisect=False
    )
    # pyformat: disable
    expected_element = [0, 0, 0,  # element 0
                        1,  # element 1
                        0, 0, 0,  # element 0
                        1]  # element 1
    expected_clip = [0, 1, 2,  # element 0
                     0,  # element 1
                     0, 1, 2,  # element 0
                     0]  # element 1
    # pyformat: enable
    self.assert_expected_metadata(
        actual_metadata,
        expected_indices(shard_options, len(actual_metadata)),
        [i % sum(actual_clip_map) for i in range(sum(actual_clip_map) * 2)],
        expected_element,
        expected_clip,
    )

  def test_repr(self):
    clip_map = [1, 7, 2, 7, 3, 5, 4, 4]
    clip_map_repr = "[1, 7, 2, 7, 3, 5, 4, 4]"
    start_index_ordered_repr = "array([ 0,  1,  8, 10, 17, 20, 25, 29, 33])"
    shard_options = sharding.ShardOptions(shard_index=0, shard_count=2)
    shard_options_repr = (
        "ShardOptions(shard_index=0, shard_count=2, drop_remainder=False)"
    )
    batch_size = 4
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=clip_map,
        shard_options=shard_options,
        batch_size=batch_size,
        num_epochs=2,
    )
    # With shard options and batch size we will use
    # BatchedContinualSequenceSampler.
    expected_sampler_repr = (
        "BatchedContinualSequenceSampler("
        f"clip_map={clip_map_repr}, shard_options={shard_options_repr}, "
        "shuffle_dataset=False, num_epochs=2, seed=0, batch_size=4)"
    )
    expected_wrapped_sampler_repr = (
        f"SamplerWrapper(sampler={expected_sampler_repr}, "
        f"start_index_ordered={start_index_ordered_repr}, seed=0)"
    )
    self.assertEqual(repr(sampler), expected_wrapped_sampler_repr)

  def test_batching(self):
    clip_map = [1, 7, 2, 7, 3, 5, 4, 4]
    # Shard 0 will have batch elements 0 and 1 which get the largest and second
    # largest element.
    shard_options = sharding.ShardOptions(shard_index=0, shard_count=2)
    batch_size = 4
    sampler = continual_sequence_sampler.get_sampler(
        clip_map=clip_map,
        shard_options=shard_options,
        batch_size=batch_size,
        num_epochs=2,
    )
    # Each shard has 2 batch elements
    sub_sampler = typing.cast(
        continual_sequence_sampler.BatchedContinualSequenceSampler,
        sampler._sampler,
    )
    self.assertLen(sub_sampler._batch_idx_sampler, 2)
    actual_clip_map = sub_sampler._batch_idx_sampler[0]._clip_map
    self.assertSequenceEqual(actual_clip_map, [2, 7])
    actual_clip_map = sub_sampler._batch_idx_sampler[1]._clip_map
    self.assertSequenceEqual(actual_clip_map, [1, 7])
    actual_metadata = _get_all_metadata(
        sampler,
        index_generator(shard_options),
        uses_bisect=False,
    )
    # Elements should alternate between the 2 batch elements.
    interleave = lambda x, y: list(itertools.chain.from_iterable(zip(x, y)))
    # There is 1 more clip per epoch in batch element 0 than in batch element 1.
    # Therefore we interleave the full 2 epochs of element 1 with 2 epochs of
    # element 0 minus the final 2 clips.
    # However, an added complication is that the sampler will not raise an index
    # error until it tries to sample an element from the 3rd epoch which happens
    # when it samples from batch element 1.
    # Therefore we expect to receive 1 more clip from the second epoch for batch
    # element 0 before raising the error.
    combine_batch_elements = lambda x, y: interleave(x[:-2], y) + [x[-2]]
    # pyformat: disable
    expected_element0 = [2, 2, 3, 3, 3, 3, 3, 3, 3,  # epoch 0
                         2, 2, 3, 3, 3, 3, 3, 3, 3]  # epoch 1
    expected_element1 = [0, 1, 1, 1, 1, 1, 1, 1,  # epoch 0
                         0, 1, 1, 1, 1, 1, 1, 1]  # epoch 1
    expected_element = combine_batch_elements(
        expected_element0,
        expected_element1,
    )
    expected_clip0 = [0, 1, 0, 1, 2, 3, 4, 5, 6,  # epoch 0
                      0, 1, 0, 1, 2, 3, 4, 5, 6]  # epoch 1
    expected_clip1 = [0, 0, 1, 2, 3, 4, 5, 6,  # epoch 0
                      0, 0, 1, 2, 3, 4, 5, 6]  # epoch 1
    expected_clip = combine_batch_elements(expected_clip0, expected_clip1)
    expected_record_key0 = [8, 9, 10, 11, 12, 13, 14, 15, 16,  # epoch 0
                            8, 9, 10, 11, 12, 13, 14, 15, 16]  # epoch 1
    expected_record_key1 = [0, 1, 2, 3, 4, 5, 6, 7,  # epoch 0
                            0, 1, 2, 3, 4, 5, 6, 7]  # epoch 1
    expected_record_key = combine_batch_elements(
        expected_record_key0,
        expected_record_key1,
    )
    # pyformat: enable
    self.assert_expected_metadata(
        actual_metadata,
        expected_indices(shard_options, len(actual_metadata)),
        expected_record_key,
        expected_element,
        expected_clip,
    )

  def test_sharding_with_shuffling(self):
    clip_map = [1, 2, 3, 4]
    shard_options = sharding.ShardOptions(shard_index=1, shard_count=2)
    expected_shard_clip_map = [2, 3]
    # pyformat: disable
    shuffled_order = [1, 0,  # epoch 0
                      0, 1]  # epoch 1
    # pyformat: enable
    fake_shuffled = FakeShuffledDataset(
        shuffled_order, len(expected_shard_clip_map)
    )
    fake_shuffled_constructor = lambda *args, **kwargs: fake_shuffled
    with mock.patch.object(
        shuffle,
        "ShuffleLazyMapDataset",
        new_callable=lambda: fake_shuffled_constructor,
    ):
      sampler = continual_sequence_sampler.get_sampler(
          clip_map=clip_map,
          shard_options=shard_options,
          num_epochs=2,
          shuffle_dataset=True,
      )
    sub_sampler = typing.cast(
        continual_sequence_sampler.BatchedContinualSequenceSampler,
        sampler._sampler,
    )
    self.assertLen(sub_sampler._batch_idx_sampler, 1)
    actual_clip_map = sub_sampler._batch_idx_sampler[0]._clip_map
    self.assertSequenceEqual(actual_clip_map, expected_shard_clip_map)
    actual_metadata = _get_all_metadata(
        sampler,
        index_generator(shard_options),
        uses_bisect=False,
    )
    # pyformat: disable
    expected_element = [2, 2, 2,  # element 2
                        1, 1,  # element 1
                        1, 1,  # element 1
                        2, 2, 2]  # element 2
    expected_clip = [0, 1, 2,  # element 2
                     0, 1,  # element 1
                     0, 1,  # element 1
                     0, 1, 2]  # element 2
    expected_record_key = [3, 4, 5,  # element 2
                           1, 2,  # element 1
                           1, 2,  # element 1
                           3, 4, 5]  # element 2
    # pyformat: enable
    self.assert_expected_metadata(
        actual_metadata,
        expected_indices(shard_options, len(actual_metadata)),
        expected_record_key,
        expected_element,
        expected_clip,
    )


if __name__ == "__main__":
  absltest.main()
