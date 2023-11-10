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
import copy
from typing import Any

from grain._src.core import tree
from grain._src.python.lazy_dataset import lazy_dataset
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import numpy as np


# SingleBinPackLazyDatasetIterator's state is defined by the a state of the
# `parent` (at which SingleBinPackLazyDatasetIterator had empty buffers) and the
# number of __next__() calls. LazyDatasetIterators should have small state that
# is cheap to serialize but doing it on every __next__() call still poses a
# performance risk. Only taking the state every N steps mostly avoids this and
# still allow for relatively fast restores of the state.
# The very is just a hint, the actually number of __next__() calls might be
# slightly higher to avoid checkpointing internal buffers.
_RECACHE_PARENT_STATE_EVERY_N = 100


@lazy_dataset.lazy_iter_dataset_function("single_bin_pack")
class SingleBinPackLazyIterDataset(lazy_dataset.LazyIterDataset):
  """Packs potentially multiple examples from the parent into a single example.

  This is one implementation of the often called "example packing"
  transformation. The parent produces examples that contain sequences of
  variable length and this transformations outputs examples of the desired
  sequence length while minizing the amound of padding.

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
      parent: lazy_dataset.LazyIterDataset,
      length_struct: PyTree[int | None],
  ):
    super().__init__(parent)
    self._length_struct = length_struct

  def __iter__(self) -> lazy_dataset.LazyDatasetIterator:
    return SingleBinPackLazyDatasetIterator(
        iter(self._parent), self._length_struct
    )  # pytype: disable=wrong-arg-types


class SingleBinPackLazyDatasetIterator(lazy_dataset.LazyDatasetIterator):
  """See SingleBinPackLazyIterDataset.

  Warning: This object is not threadsafe!
  """

  def __init__(
      self,
      parent: lazy_dataset.LazyDatasetIterator,
      length_struct: PyTree[int | None],
  ):
    self._parent = parent
    self._length_struct = length_struct
    # Same as above but flattened. Some operations are easier using the
    # flattened representation.
    self._flat_lengths: list[int | None] = tree.flatten(length_struct)
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
    while True:
      # We got fully packed examples in the buffer.
      if self._packed_elements:
        return self._packed_elements.popleft()

      try:
        flat_element = self._get_next_from_parent()
      except StopIteration as e:
        # Parent iterator exhausted. Yield whatever is in the buffer as last
        # (potentially heavily padded) element.
        if self._element_buffer:
          packed_element = self._pack_elements(self._element_buffer)
          self._element_buffer = []
          self._element_buffer_space = copy.copy(self._flat_lengths)
          return packed_element
        else:
          raise e
      if self._is_fully_packed(flat_element):
        # To avoid uncessary splitting/truncation we pass through examples that
        # seem to be fully packed. This does change the order of examples but
        # is allowed and handled correctly by our checkpointing.
        # This behaviour is especially important when examples are splits of
        # larger examples that already have the desired length.
        self._packed_elements.append(self._pack_elements([flat_element]))
        continue

      # Concat element to incomplete_element.
      is_fully_packed = self._append_to_next_element(flat_element)
      if is_fully_packed:
        self._packed_elements.append(self._pack_elements(self._element_buffer))
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
