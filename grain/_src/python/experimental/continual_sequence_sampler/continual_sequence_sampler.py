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
"""Create an index sampler for sequential batches.

This deals with the scenario where we have a dataset with n "elements" each
consisting of a variable number of "clips" of fixed size.

A record_key refers to a single clip within an element.

Given an index, we convert this into an index within the epoch and we can use
this as a record_key.
We can convert the record_key to an element and a clip within the element.
i.e. given elements which have the following number of clips:
  clip_map = [3, 1, 5]
then the start clip map will be computed as the following:
  _start_index = [0, 3, 4, 9] (the final entry is the total number of clips)
We can then use this info to compute an element index and how many clips through
the corresponding element we are (called clip_id).
If we go through the dataset sequentially this will be computed in constant
time, otherwise this will take O(log(#elements)) time.
"""

import bisect
from collections.abc import Sequence
import dataclasses
from typing import Optional

from grain._src.python import record
import numpy as np


@dataclasses.dataclass(frozen=True)
class ElementClip:
  element: int
  clip: int


def _element_clip_from_index(
    index_within_epoch: int,
    start_index: np.ndarray,
    current_element_index: int,
) -> ElementClip:
  """Get the element id and clip id for the given index and epoch.

  Args:
    index_within_epoch: The index within the epoch.
    start_index: The start index for each element.
    current_element_index: A guess for which element the index is in. If this is
      correct we can compute the element index and clip id in constant time
      rather than logarithmic time. If we are reading data sequentially we use
      this to avoid ever having to compute in logarithmic time.

  Returns:
    A dataclass containing the following:
      - The element index.
      - The clip index within the element.
  """

  # Now, check if we are still within the same element as before.
  current_element_start = start_index[current_element_index]
  next_element_start = start_index[current_element_index + 1]
  if current_element_start <= index_within_epoch < next_element_start:
    clip_id = index_within_epoch - current_element_start
    return ElementClip(element=current_element_index, clip=clip_id)
  # Otherwise check if we have started a new epoch.
  elif index_within_epoch == 0:
    current_element_index = 0
  # In the normal case we will just move to the next element.
  elif index_within_epoch == next_element_start:
    current_element_index += 1
  # Otherwise we must compute which element we are in.
  else:
    current_element_index = (
        bisect.bisect_left(start_index, index_within_epoch + 1) - 1
    )
  clip_id = index_within_epoch - start_index[current_element_index]
  return ElementClip(element=current_element_index, clip=clip_id)


class ContinualSequenceSampler:
  """Sample clips from elements in a continual sequence."""

  def __init__(
      self,
      clip_map: Sequence[int],
      num_epochs: Optional[int] = None,
      seed: Optional[int] = None,
  ):
    self._seed = seed
    if not clip_map:
      raise ValueError("The index is empty after applying filters.")

    self._current_element_index = 0
    self._start_index = np.cumsum([0] + list(clip_map))
    self._max_index = (
        None if num_epochs is None else num_epochs * self._start_index[-1]
    )

  @property
  def current_element_index(self) -> int:
    return self._current_element_index

  def set_element_clip_from_index(self, index: int) -> ElementClip:
    """Set the element id for the given index and return this and the clip."""
    if self._max_index is not None and index >= self._max_index:
      raise IndexError(
          "RecordMetadata object index is out of bounds; Got index "
          f"{index}, allowed indices should be in [0, {self._max_index}]"
      )
    # First get the index within the current epoch.
    index_within_epoch = index % self._start_index[-1]
    element_clip = _element_clip_from_index(
        index_within_epoch,
        self._start_index,
        self._current_element_index,
    )
    self._current_element_index = element_clip.element
    return element_clip

  def __getitem__(self, index: int) -> record.RecordMetadata:
    element_clip = self.set_element_clip_from_index(index)
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    next_record = record.RecordMetadata(
        index=index,
        record_key=self._start_index[element_clip.element] + element_clip.clip,
        rng=rng,
    )
    return next_record

  def record_key_to_element_and_clip(self, record_key: int) -> ElementClip:
    """Convert a record key to an element index and a clip index."""
    # The record key refers to the position within the entire dataset, we
    # use the helper function to compute which element and which clip within the
    # element this maps to.
    return _element_clip_from_index(
        record_key,
        self._start_index,
        self._current_element_index,
    )


def get_sampler(
    clip_map: Sequence[int],
    seed: int = 0,
    **kwargs,
) -> ContinualSequenceSampler:
  """Get a continual sequence sampler."""
  return ContinualSequenceSampler(clip_map, seed=seed, **kwargs)
