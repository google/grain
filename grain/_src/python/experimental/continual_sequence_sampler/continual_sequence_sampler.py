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

The top-level elements may be shuffled, but the clips within each element must
be returned in order. The shuffled order of elements is stored in
ContinualSequenceGenerator in _element_index.

Elements are split across shards and batch elements and the top level index is
converted to an index for the current shard and batch element.

Given an index, we convert this into an epoch number and an index within the
epoch. When the epoch changes we recompute the start clip for each element.
i.e. given a shard and batch element with elements which have the following
number of clips:
  clip_map = [3, 1, 5]
  _element_index = [2, 0, 1] (for example)
then the start clip map will be computed as the following:
  _start_index = [0, 5, 8, 9] (the final entry is the total number of clips)
We can then use this info to compute an index into _element_index and how many
clips through the corresponding element we are (called clip_id).
If we go through the dataset sequentially this will be computed in constant
time, otherwise this will take O(log(#elements)) time.
We can then convert this into a record key which is an index into the unshuffled
dataset.
e.g. in the example above given the index 7 we would compute from _start_index
that the index into _element_index is 1 (with clip_id=2). We would find the
corresponding element is 0 and so the record_key is 0 + clip_id = 2.
The whole first epoch would give record_keys = [4, 5, 6, 7, 8, 0, 1, 2, 3].
These record keys can be converted back to an element index and clip id using
the function "get_element_clip_from_record_key".
"""

import bisect
from collections.abc import Sequence
import dataclasses
import heapq
from typing import Optional, Tuple

from grain._src.core import sharding
from grain._src.python import record
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import shuffle
import numpy as np


@dataclasses.dataclass(frozen=True)
class ElementClip:
  element: int
  clip: int


def shard_elements(
    clip_map: Sequence[int], index: int, size: int
) -> Tuple[Sequence[int], Sequence[int]]:
  """Shard elements attempting to balance the number of clips in each shard.

  This uses the Longest Processing Time First scheduling algorithm to try to
  balance the clips across batch elements.

  Args:
    clip_map: The number of clips in each element.
    index: The index of the shard we are using.
    size: The total number of shards.

  Returns:
    A tuple containing the following:
      - The number of clips for elements in this shard.
      - A map which converts an index into the new clip map to an index into the
        clip map passed in.
  """
  per_shard_size_heap = [(0, i) for i in range(size)]
  new_clip_map = []
  descending_clip_lengths = sorted(
      [(element_length, i) for i, element_length in enumerate(clip_map)],
      reverse=True,
  )
  for element_length, i in descending_clip_lengths:
    smallest_end_time, smallest_index = heapq.heappop(per_shard_size_heap)
    heapq.heappush(
        per_shard_size_heap,
        (smallest_end_time + element_length, smallest_index),
    )
    if smallest_index == index:
      new_clip_map.append((i, element_length))
  # Sort by the original index to retain the original order.
  new_clip_map = sorted(new_clip_map)
  original_indices, lengths = zip(*new_clip_map)
  return lengths, original_indices


def _get_shuffled_element_index(
    element_idx_in_epoch: int,
    epoch: int,
    element_index: Sequence[int],
) -> int:
  num_elements = len(element_index)
  # If epoch > 0, then we index past the length of element_index. This works
  # with lazy datasets.
  assert epoch == 0 or isinstance(element_index, lazy_dataset.LazyMapDataset)
  return element_index[epoch * num_elements + element_idx_in_epoch]


def _element_clip_from_index(
    index_within_epoch: int,
    epoch: int,
    start_index: np.ndarray,
    element_index: Sequence[int],
    current_element_index: int,
) -> Tuple[ElementClip, int]:
  """Get the element id and clip id for the given index and epoch.

  Args:
    index_within_epoch: The index within the epoch.
    epoch: Which epoch we are on.
    start_index: The start index for each element.
    element_index: The sequence in which to process elements.
    current_element_index: A guess for which element the index is in. If this is
      correct we can compute the element index and clip id in constant time
      rather than logarithmic time. If we are reading data sequentially we use
      this to avoid ever having to compute in logarithmic time.

  Returns:
    A tuple containing the following:
      - A dataclass containing:
        - The element index corresponding to the index passed in, respecting the
          order in the clip map
        - The clip index within the element.
      - An update to the current element index into the element_index sequence.
  """
  # Now, check if we are still within the same element as before.
  current_element_start = start_index[current_element_index]
  next_element_start = start_index[current_element_index + 1]
  if current_element_start <= index_within_epoch < next_element_start:
    # Get the shuffled element index
    element_id = _get_shuffled_element_index(
        current_element_index, epoch, element_index
    )
    clip_id = index_within_epoch - current_element_start
    return ElementClip(element=element_id, clip=clip_id), current_element_index
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
  # Get the shuffled element index
  element_id = _get_shuffled_element_index(
      current_element_index, epoch, element_index
  )
  clip_id = index_within_epoch - start_index[current_element_index]
  return ElementClip(element=element_id, clip=clip_id), current_element_index


class ContinualSequenceSampler:
  """Sample clips from elements in a continual sequence."""

  def __init__(
      self,
      clip_map: Sequence[int],
      shuffle_dataset: bool = False,
      num_epochs: Optional[int] = None,
      seed: Optional[int] = None,
  ):
    self._seed = seed
    if not clip_map:
      raise ValueError("The index is empty after applying filters.")

    self._clip_map = clip_map

    self._element_index = lazy_dataset.RangeLazyMapDataset(len(clip_map))
    self._shuffle_dataset = shuffle_dataset
    if shuffle_dataset:
      self._element_index = shuffle.ShuffleLazyMapDataset(
          self._element_index, seed=seed
      )

    self._current_element_index = 0
    self._current_epoch = 0
    self._start_index = self._compute_start_index_tree(0)

    self._max_index = (
        None if num_epochs is None else num_epochs * self._start_index[-1]
    )

  def __repr__(self) -> str:
    return (
        f"ContinualSequenceSampler(clip_map={self._clip_map!r}, "
        f"shuffle_dataset={self._shuffle_dataset!r}, "
        f"num_epochs={self._num_epochs!r}, "
        f"seed={self._seed!r})"
    )

  def _in_epoch(self, idx: int) -> Tuple[int, int]:
    """Get the index within the current epoch and the current epoch."""
    return idx % self._start_index[-1], idx // self._start_index[-1]

  def _maybe_compute_start_index_tree(self, epoch: int) -> None:
    # We must recompute the start index per element each epoch because it
    # depends on the order of elements which changes each epoch.
    if self._current_epoch == epoch:
      return
    self._start_index = self._compute_start_index_tree(epoch)
    self._current_epoch = epoch

  def _compute_start_index_tree(self, epoch: int) -> np.ndarray:
    """Compute the start index tree for the given epoch."""
    shuffled_indices = [
        _get_shuffled_element_index(i, epoch, self._element_index)
        for i in range(len(self._clip_map))
    ]
    shuffled_clip_map = np.take(self._clip_map, shuffled_indices)
    return np.cumsum(np.concatenate(([0], shuffled_clip_map)))

  @property
  def current_element_index(self) -> int:
    # Set the element index to the most recently set element index in
    # __getitem__, converted to an index into the ordered start index map.
    current_element_index = self._current_element_index
    return _get_shuffled_element_index(
        current_element_index, self._current_epoch, self._element_index
    )

  def set_element_clip_from_index(self, index: int) -> ElementClip:
    """Set the element id for the given index and return this and the clip."""
    if self._max_index is not None and index >= self._max_index:
      raise IndexError(
          "RecordMetadata object index is out of bounds; Got index "
          f"{index}, allowed indices should be in [0, {self._max_index}]"
      )
    # First get the index within the current epoch.
    index_within_epoch, epoch = self._in_epoch(index)
    # Ensure the start index map is correct for the epoch we are in.
    self._maybe_compute_start_index_tree(epoch)
    element_clip, self._current_element_index = _element_clip_from_index(
        index_within_epoch,
        epoch,
        self._start_index,
        self._element_index,
        self._current_element_index,
    )
    return element_clip


class BatchedContinualSequenceSampler:
  """Sample clips from elements in a continual sequence sharding into batch elements."""

  def __init__(
      self,
      clip_map: Sequence[int],
      shard_options: sharding.ShardOptions,
      shuffle_dataset: bool = False,
      num_epochs: Optional[int] = None,
      seed=0,
      batch_size: Optional[int] = None,
  ):
    if len(clip_map) < shard_options.shard_count:
      raise ValueError(
          "Cannot shard with fewer videos than shards. "
          f"Num videos = {len(clip_map)}, "
          f"num shards = {shard_options.shard_count}."
      )
    self._seed = seed
    self._start_index_ordered = np.cumsum([0] + list(clip_map))
    self._batch_size = (
        batch_size if batch_size is not None else shard_options.shard_count
    )
    if self._batch_size % shard_options.shard_count != 0:
      raise ValueError(
          "batch_size must be a multiple of shard_count. "
          f"batch_size = {batch_size}, "
          f"shard_count = {shard_options.shard_count}."
      )
    per_shard_bs = self._batch_size // shard_options.shard_count
    self._invert_batch_idx = []
    self._batch_idx_sampler = []
    self._shard_options = shard_options
    self._shuffle_dataset = shuffle_dataset
    self._num_epochs = num_epochs
    self._clip_map = clip_map
    for i in range(
        shard_options.shard_index * per_shard_bs,
        (shard_options.shard_index + 1) * per_shard_bs,
    ):
      batch_clips, invert_batch_idx = shard_elements(
          clip_map, i, self._batch_size
      )
      self._invert_batch_idx.append(invert_batch_idx)
      self._batch_idx_sampler.append(
          ContinualSequenceSampler(
              batch_clips, shuffle_dataset, num_epochs, seed + i
          )
      )
    self._current_batch_element = 0

  def __repr__(self) -> str:
    return (
        f"BatchedContinualSequenceSampler(clip_map={self._clip_map!r}, "
        f"shard_options={self._shard_options!r}, "
        f"shuffle_dataset={self._shuffle_dataset!r}, "
        f"num_epochs={self._num_epochs!r}, "
        f"seed={self._seed!r}, "
        f"batch_size={self._batch_size!r})"
    )

  def get_element_clip_from_record_key(
      self,
      record_key: int,
  ) -> ElementClip:
    """Get the element id and clip id for the given record key."""
    element_clip, _ = _element_clip_from_index(
        record_key,
        0,
        self._start_index_ordered,
        list(range(len(self._start_index_ordered))),
        self.current_element_index,
    )
    return element_clip

  def set_element_clip_from_index(self, index: int) -> ElementClip:
    """Set the element id for the given index and return this and the clip."""
    per_shard_index = index // self._shard_options.shard_count
    per_shard_bs = self._batch_size // self._shard_options.shard_count
    sub_index = per_shard_index // per_shard_bs
    self._current_batch_element = per_shard_index % per_shard_bs
    sampler = self._batch_idx_sampler[self._current_batch_element]
    invert_batch_idx = self._invert_batch_idx[self._current_batch_element]
    element_clip = sampler.set_element_clip_from_index(sub_index)
    original_element = invert_batch_idx[element_clip.element]
    return ElementClip(element=original_element, clip=element_clip.clip)

  @property
  def current_element_index(self) -> int:
    sampler = self._batch_idx_sampler[self._current_batch_element]
    invert_batch_idx = self._invert_batch_idx[self._current_batch_element]
    return invert_batch_idx[sampler.current_element_index]


class SamplerWrapper:
  """Wraps a ContinualSequenceSampler to conform to the Sampler protocol."""

  def __init__(
      self,
      sampler: ContinualSequenceSampler | BatchedContinualSequenceSampler,
      start_index_ordered: np.ndarray,
      seed: int,
  ):
    self._sampler = sampler
    self._start_index_ordered = start_index_ordered
    self._seed = seed

  def __getitem__(self, index: int) -> record.RecordMetadata:
    element_clip = self._sampler.set_element_clip_from_index(index)
    original_start_index = self._start_index_ordered[element_clip.element]
    rng = None
    if self._seed is not None:
      rng = np.random.Generator(np.random.Philox(key=self._seed + index))
    next_record = record.RecordMetadata(
        index=index,
        record_key=original_start_index + element_clip.clip,
        rng=rng,
    )
    return next_record

  def __repr__(self) -> str:
    return (
        f"SamplerWrapper(sampler={self._sampler!r}, "
        f"start_index_ordered={self._start_index_ordered!r}, "
        f"seed={self._seed!r})"
    )

  def record_key_to_element_and_clip(self, record_key: int) -> ElementClip:
    """Convert a record key to an element index and a clip index."""
    # The record key refers to the position within the entire dataset, we
    # use the helper function to compute which element and which clip within the
    # element this maps to.
    element_clip, _ = _element_clip_from_index(
        record_key,
        0,
        self._start_index_ordered,
        list(range(len(self._start_index_ordered))),
        self._sampler.current_element_index,
    )
    return element_clip


def get_sampler(
    clip_map: Sequence[int],
    seed: int = 0,
    shard_options: Optional[sharding.ShardOptions] = None,
    shuffle_dataset: bool = False,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> SamplerWrapper:
  """Get a continual sequence sampler."""
  if shard_options is not None:
    sampler = BatchedContinualSequenceSampler(
        clip_map,
        shard_options,
        shuffle_dataset=shuffle_dataset,
        num_epochs=num_epochs,
        seed=seed,
        batch_size=batch_size,
    )
  else:
    if batch_size is not None:
      raise ValueError(
          "shard_options must be specified if batch_size is specified"
      )
    sampler = ContinualSequenceSampler(
        clip_map,
        shuffle_dataset=shuffle_dataset,
        num_epochs=num_epochs,
        seed=seed,
    )
  wrapper = SamplerWrapper(sampler, np.cumsum([0] + list(clip_map)), seed)
  return wrapper
