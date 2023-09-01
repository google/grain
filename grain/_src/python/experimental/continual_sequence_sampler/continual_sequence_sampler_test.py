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
from typing import Iterable, Sequence
from unittest import mock

from absl.testing import absltest
from grain._src.python.experimental.continual_sequence_sampler import continual_sequence_sampler


@dataclasses.dataclass(frozen=True)
class RecordMetadata:
  index: int
  record_key: int
  element: int
  clip: int


def _get_all_metadata(
    sampler: continual_sequence_sampler.ContinualSequenceSampler,
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


if __name__ == "__main__":
  absltest.main()
