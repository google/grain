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
"""Implements packing transformations."""
import collections
from collections.abc import Sequence
import copy
from typing import Any, Optional
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import packing_packed_batch
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np

import tree


# SingleBinPackDatasetIterator's state is defined by the a state of the
# `parent` (at which SingleBinPackDatasetIterator had empty buffers) and the
# number of __next__() calls. DatasetIterators should have small state that
# is cheap to serialize but doing it on every __next__() call still poses a
# performance risk. Only taking the state every N steps mostly avoids this and
# still allow for relatively fast restores of the state.
# The very is just a hint, the actually number of __next__() calls might be
# slightly higher to avoid checkpointing internal buffers.
_RECACHE_PARENT_STATE_EVERY_N = 100


class SingleBinPackIterDataset(dataset.IterDataset):
  """Packs potentially multiple examples from the parent into a single example.

  This is one implementation of the often called "example packing"
  transformation. The parent produces examples that contain sequences of
  variable length and this transformations outputs examples of the desired
  sequence length while minizing the amount of padding.

  # Properties of SingleBinPack:
  SingleBinPack packs a single example at a time and passes through any examples
  that seem already packed.
  When only single features is packed this produces 0 padding. However some
  examples will be truncated. And when the parent iterator is exhausted the
  remaining examples in the buffer will be packed together and padded to form
  a full packed example.

  # Details
  The `length_struct` should have the same structure as examples from the
  parent and provide is desired sequence length. The sequence length of
  a features is the size of its first dimension. If the desired sequence
  length is `None` the feature will not be packed and the output will just
  contain a list of the features the individual examples.

  Warning: Make sure that examples from the `parent` have the same structure
  as `length_struct`. Python containers such as lists are not considered leafs -
  NumPy array are.

  For packed features the output will have the desired feature length by
  combining multiple examples. If there is only a single packed feature it is
  guarantee to not contain padding.
  Warning: When combining examples that add up to a sequence longer than the
  target sequence length the remainder of the last example will be dropped!
  (This could be changed by improving the implementation.)

  As common each packed feature there will have 3 outputs:
  - The concatenated values.
  - The segmentation that indicate the different examples.
  - The positions of values within each example.

  If the input is a flat dictionaries segmentations and positions will be added
  as new entries. Otherwise the output will contain tuples (value, segmentation,
  positions).
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      length_struct: PyTree[Optional[int]],
  ):
    super().__init__(parent)
    self._length_struct = length_struct

  def __str__(self) -> str:
    return "SingleBinPackIterDataset"

  def __iter__(self) -> dataset.DatasetIterator:
    return SingleBinPackDatasetIterator(
        self._parent.__iter__(), self._length_struct, self._stats
    )


class SingleBinPackDatasetIterator(dataset.DatasetIterator):
  """See SingleBinPackIterDataset.

  Warning: This object is not threadsafe!
  """

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      length_struct: PyTree[Optional[int]],
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._length_struct = length_struct
    # Same as above but flattened. Some operations are easier using the
    # flattened representation.
    self._flat_lengths: list[Optional[int]] = tree.flatten(length_struct)
    # Buffer for fully packed elements (not flattened)
    self._packed_elements = collections.deque()
    # Variable length list of flat elements going into the next packed example.
    self._element_buffer = []
    # List with space remaining per feature.
    self._element_buffer_space = copy.copy(self._flat_lengths)
    # State of parent iterator when the buffers where empty and num_next_calls
    # was 0.
    self._last_parent_state: dict[str, Any] = self._parent.get_state()
    # Number of calls to __next__() since we cached the parent state.
    # We refresh the _last_parent_state and reset _num_next_calls when
    # buffer are empty.
    self._num_next_calls = 0

  def _get_next_from_parent(self) -> tuple[int, list[Any]]:
    if self._num_next_calls > _RECACHE_PARENT_STATE_EVERY_N:
      # Update `_last_state` only if buffers are empty.
      if not self._packed_elements and not self._element_buffer:
        self._last_parent_state = self._parent.get_state()
        self._num_next_calls = 0

    element = next(self._parent)
    flat_element = tree.flatten(element)
    return flat_element

  def __next__(self):
    self._num_next_calls += 1
    timer = dataset_stats.Timer()
    while True:
      # We got fully packed examples in the buffer.
      if self._packed_elements:
        with self._stats.record_self_time(offset_ns=timer.value()):
          result = self._packed_elements.popleft()
        return self._stats.record_output_spec(result)

      try:
        flat_element = self._get_next_from_parent()
      except StopIteration as e:
        # Parent iterator exhausted. Yield whatever is in the buffer as last
        # (potentially heavily padded) element.
        if self._element_buffer:
          with self._stats.record_self_time(offset_ns=timer.value()):
            packed_element = self._pack_elements(self._element_buffer)
            self._element_buffer = []
            self._element_buffer_space = copy.copy(self._flat_lengths)
          return self._stats.record_output_spec(packed_element)
        else:
          raise e
      with timer:
        if self._is_fully_packed(flat_element):
          # To avoid uncessary splitting/truncation we pass through examples
          # that seem to be fully packed. This does change the order of examples
          # but is allowed and handled correctly by our checkpointing.
          # This behaviour is especially important when examples are splits of
          # larger examples that already have the desired length.
          self._packed_elements.append(self._pack_elements([flat_element]))
          continue

        # Concat element to incomplete_element.
        is_fully_packed = self._append_to_next_element(flat_element)
        if is_fully_packed:
          self._packed_elements.append(
              self._pack_elements(self._element_buffer)
          )
          self._element_buffer = []
          self._element_buffer_space = copy.copy(self._flat_lengths)

  def _is_fully_packed(self, flat_element):
    return any(
        target_length is not None and len(y) >= target_length
        for target_length, y in zip(self._flat_lengths, flat_element)
    )

  def _append_to_next_element(self, flat_element) -> bool:
    self._element_buffer.append(flat_element)
    is_fully_packed = False
    for i in range(len(self._flat_lengths)):
      if self._flat_lengths[i] is None:
        continue
      if len(flat_element[i]) >= self._element_buffer_space[i]:
        self._element_buffer_space[i] = 0
        is_fully_packed = True
      else:
        self._element_buffer_space[i] -= len(flat_element[i])
    return is_fully_packed

  def _pack_elements(self, flat_elements: list[Any]):
    flat_packed_element = []
    for feature in range(len(self._flat_lengths)):
      if self._flat_lengths[feature] is None:
        # Feature should not be packed.
        flat_packed_element.append(
            [flat_elements[i][feature] for i in range(len(flat_elements))]
        )
        continue
      sequence_length = self._flat_lengths[feature]
      remaining_dims = flat_elements[0][feature].shape[1:]
      shape = [sequence_length, *remaining_dims]
      dtype = flat_elements[0][feature].dtype
      values = np.zeros(shape, dtype=dtype)
      segmentations = np.zeros(shape=[sequence_length], dtype=np.int32)
      positions = np.zeros(shape=[sequence_length], dtype=np.int32)

      start = 0
      for i in range(len(flat_elements)):
        length = min(len(flat_elements[i][feature]), sequence_length - start)
        end = start + length
        values[start:end] = flat_elements[i][feature][:length]
        segmentations[start:end] = i + 1
        positions[start:end] = np.arange(length)
        start += length
      flat_packed_element.append((values, segmentations, positions))
    packed_element = tree.unflatten_as(self._length_struct, flat_packed_element)
    # Special treatment for dictionaries.
    if isinstance(packed_element, dict):
      for key in list(packed_element):
        value = packed_element[key]
        if isinstance(value, tuple) and len(value) == 3:
          packed_element[key] = value[0]
          packed_element[f"{key}_segment_ids"] = value[1]
          packed_element[f"{key}_positions"] = value[2]
    return packed_element

  def get_state(self) -> dict[str, Any]:
    return {
        "last_parent_state": self._last_parent_state,
        "num_next_calls": self._num_next_calls,
    }

  def set_state(self, state: dict[str, Any]):
    self._last_parent_state = state["last_parent_state"]
    self._parent.set_state(self._last_parent_state)
    # Empty buffers.
    self._packed_elements = collections.deque()
    self._element_buffer = []
    self._element_buffer_space = copy.copy(self._flat_lengths)
    # Advance for num_next_calls.
    self._num_next_calls = 0
    # Replay pipeline for `num_next_calls`.
    for _ in range(state["num_next_calls"]):
      next(self)
    assert self._num_next_calls == state["num_next_calls"]


class FirstFitPackIterDataset(dataset.IterDataset):
  """Implements first-fit packing of sequences.

  Packing, compared to concat-and-split, avoids splitting sequences by padding
  instead. Larger number of packing bins reduce the amount of padding. If the
  number of bins is large, this can cause epoch leakage (data points from
  multiple epochs getting packed together).

  This uses a simple first-fit packing algorithm that:
  1. Creates N bins.
  2. Adds elements (in the order coming from the parent) to the first bin that
  has enough space.
  3. Once an element doesn't fit, emits all N bins as elements.
  4. (optional) Shuffles bins.
  5. Loops back to 1 and starts with the element that didn't fit.

  This iterator is easy to make deterministic, but it has the downside that some
  bins (usually the bottom bins) have a lot of padding. To avoid this pattern,
  we add an option to shuffle the bins before emitting.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      length_struct: Any,
      num_packing_bins: int,
      shuffle_bins: bool = True,
      meta_features: Sequence[str] = (),
  ):
    """Creates a dataset that packs sequences from the parent dataset.

    Args:
      parent: Parent dataset with variable length sequences. Sequence cannot be
        longer than their length_struct value.
      length_struct: Target sequence length for each feature.
      num_packing_bins: Number of bins to pack sequences into.
      shuffle_bins: Whether to shuffle bins after packing.
      meta_features: Meta features that do not need *_segment_ids and
        *_positions features.
    """
    super().__init__(parent)
    self._length_struct = length_struct
    self._num_packing_bins = num_packing_bins
    self._shuffle_bins = shuffle_bins
    self._meta_features = meta_features

  def __str__(self) -> str:
    return "FirstFitPackIterDataset"

  def __iter__(self) -> dataset.DatasetIterator:
    return FirstFitPackDatasetIterator(
        self._parent.__iter__(),
        num_packing_bins=self._num_packing_bins,
        length_struct=self._length_struct,
        shuffle_bins=self._shuffle_bins,
        meta_features=self._meta_features,
        stats=self._stats,
    )


class FirstFitPackDatasetIterator(dataset.DatasetIterator):
  """Iterator for the first-fit packing transformation."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      num_packing_bins: int,
      length_struct: PyTree[Optional[int]],
      shuffle_bins: bool,
      meta_features: Sequence[str],
      stats: dataset_stats.Stats,
  ):
    super().__init__(stats)
    self._parent = parent
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._shuffle_bins = shuffle_bins
    self._meta_features = meta_features
    self._reset()

  def _reset(self):
    self._current_batch = None  # Not yet fully packed.
    self._current_batch_parent_state = self._parent.get_state()
    # If available, fully packed but rows [:self._next_row] were already
    # emitted.
    self._packed_batch = None
    # The last packed batch can be partial and have few bins with elements.
    self._packed_batch_num_bins = None
    self._packed_batch_parent_state = None
    # _next_row gets reset between batches.
    # _counter is a global counter for rows emitted, does not get reset.
    self._next_row = 0
    self._counter = 0  # Used for RNG seed.
    self._shuffled_rows = None

  def get_state(self) -> dict[str, Any]:
    if self._packed_batch_parent_state is None:
      # If we haven't finished packing a batch or exausted the parent iterator
      # packed_batch_parent_state will be None and current_batch_parent_state
      # will point to the state before the first element in the current batch.
      parent_state = self._current_batch_parent_state
    else:
      parent_state = self._packed_batch_parent_state
    return {
        "parent": parent_state,
        "next_row": self._next_row,
        "counter": self._counter,
    }

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(state["parent"])
    self._reset()
    self._next_row = state["next_row"]
    self._counter = state["counter"]

  def _finalize_current_batch(self, element_for_shapes):
    assert self._current_batch is not None
    assert self._current_batch_parent_state is not None
    self._packed_batch = self._current_batch.get_packed_batch()
    self._packed_batch_parent_state = self._current_batch_parent_state
    # Detect number of bins. The last batch can be partial.
    self._packed_batch_num_bins = max(
        tree.flatten(
            tree.map_structure(lambda x: x.shape[0], self._packed_batch)
        )
    )
    assert self._packed_batch_num_bins <= self._num_packing_bins
    if self._shuffle_bins:
      seed = self._counter - self._next_row
      self._shuffled_rows = np.random.default_rng(seed).permuted(
          range(self._packed_batch_num_bins)
      )

    if element_for_shapes is None:
      self._current_batch = None
    else:
      self._current_batch = packing_packed_batch.PackedBatch(
          element_for_shapes,
          self._num_packing_bins,
          self._length_struct,
          meta_features=self._meta_features,
      )

  def __next__(self):
    timer = dataset_stats.Timer()
    if self._packed_batch is not None:
      with self._stats.record_self_time(offset_ns=timer.value()):
        if self._shuffle_bins:
          next_row = self._shuffled_rows[self._next_row]
        else:
          next_row = self._next_row
        element = tree.map_structure(lambda x: x[next_row], self._packed_batch)
        self._next_row += 1
        self._counter += 1
        if self._next_row >= self._packed_batch_num_bins:
          self._packed_batch = None
          self._packed_batch_parent_state = None
          self._next_row = 0
          self._shuffled_rows = None
        return self._stats.record_output_spec(element)

    while True:
      prior_iterator_state = self._parent.get_state()
      assert prior_iterator_state is not None
      try:
        element = next(self._parent)
      except StopIteration as e:
        if self._current_batch:
          with timer:
            self._finalize_current_batch(None)
            self._current_batch_parent_state = prior_iterator_state
          return next(self)
        else:
          # The inner iterator is exhausted and there is no current batch, so
          # the packed iterator is also exhausted.
          raise StopIteration() from e

      with timer:
        # Remove elements not in packing struct.
        element = tree.map_structure_up_to(
            self._length_struct, lambda x: x, element
        )

        if self._current_batch is None:  # pytype: disable=attribute-error
          # Use `element` to set dtypes + trailing dimensions.
          # We are not adding the element to the batch, just initializing it.
          self._current_batch = packing_packed_batch.PackedBatch(
              element,
              self._num_packing_bins,
              self._length_struct,
              meta_features=self._meta_features,
          )
          self._current_batch_parent_state = prior_iterator_state

        # Try adding element to the current packed batch.
        failing_components = self._current_batch.try_add_to_batch(element)

      # When we have a full batch, yield the current packed data,
      # and then start a new batch with this element.
      if failing_components is not None:
        with timer:
          self._finalize_current_batch(element)
          self._current_batch_parent_state = prior_iterator_state
          assert self._current_batch is not None

          if self._current_batch.try_add_to_batch(element) is not None:
            # If we can't pack a single example into an empty batch then we
            # can't continue at all.
            element_shape = tree.map_structure(lambda x: x.shape, element)
            raise ValueError(
                "Could not add element to empty packed batch! Packed batch has"
                f" packing sequence_lengths: {self._length_struct} while"
                f" element has shape: {element_shape}"
            )
        # We now have packed batch.
        return next(self)
