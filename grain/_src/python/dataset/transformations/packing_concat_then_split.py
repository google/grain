# Copyright 2025 Google LLC
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
"""Provides ConcatThenSplitIterDataset.

This transformations takes elements with variable length sequence features,
concatenates (over the whole dataset) them and then splits the result into
sequence of the desired sequence length. If elements contain a single sequence
feature this will pack without padding.

The algorithm concatenates elements in a buffer. When they don't fit anymore,
the remainder element is split, and the packed buffer is added to the output.
A notable optimization is that we can directly pass elements that already have
the desired sequence length to the output, as they are already packed.

If an element contains several features, we pack the limiting feature, and then
pad the other features.

The state is checkpointed everytime the buffer is empty or contains only one
remainder element. In order to restore any state, we restore to the last
checkpoint and then advance packing until the state is restored.

An example of running this algorithm for the inputs of given sizes:

r1: 3 r2: 6 r3: 2 r4: 5 r5: 6 r6: 4 - with split_full_length_features=False

--------------------------------------------------------------------------------
step:     add r1 to buffer      add r2 to output      add r3 to buffer
output:   [1, 1, 1, 0, 0, 0] -> [1, 1, 1, 0, 0, 0] -> [1, 1, 1, 3, 3, 0]
                  -             [2, 2, 2, 2, 2, 2]    [2, 2, 2, 2, 2, 2]
                  -                     -                     -
                  -                     -                     -
                  -                     -                     -
--------------------------------------------------------------------------------
step:     split r4 (size 5)     add r5 (size 6)       split r6 (size 4)
output:   [1, 1, 1, 3, 3, 4] -> [1, 1, 1, 3, 3, 4] -> [1, 1, 1, 3, 3, 4]
          [2, 2, 2, 2, 2, 2]    [2, 2, 2, 2, 2, 2]    [2, 2, 2, 2, 2, 2]
          [4, 4, 4, 4, 0, 0]    [4, 4, 4, 4, 0, 0]    [4, 4, 4, 4, 6, 6]
                  -             [5, 5, 5, 5, 5, 5]    [5, 5, 5, 5, 5, 5]
                  -                     -             [6, 6, 0, 0, 0, 0]
"""

from __future__ import annotations

import collections
from collections.abc import Collection, Mapping, Sequence
import dataclasses
import datetime
import enum
from typing import Any

from absl import logging
from grain._src.core import exceptions
from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
import numpy as np

_EMPTY_SLICE = (-1, -1)  # sentinel value for empty slices.


class BOSHandling(enum.Enum):
  """The BOS handling done inside a packing algorithm."""

  # The packing algorithm will not add or remove any BOS to/from its inputs.
  DO_NOTHING = enum.auto()
  # Replaces the first token of each sequence with BOS. If the token
  # was already BOS this is a NOOP, if the token was not BOS then that token
  # is essentially dropped.
  REPLACE_FIRST_TOKEN_WITH_BOS = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CtsConfig:
  """Config for ConcatThenSplit. See docstring of ConcatThenSplitIterDataset."""

  length_struct: Mapping[str, int]
  meta_features: Collection[str]
  split_full_length_features: bool
  bos_handling: BOSHandling
  bos_features: Collection[str]
  bos_token_id: int | None

  def __post_init__(self):
    if self.bos_handling == BOSHandling.DO_NOTHING and (
        self.bos_token_id is not None or self.bos_features
    ):
      raise ValueError(
          f"bos_handling={self.bos_handling} indicates that BOS handling is not"
          " used. So bos_token_id and bos_features should not be set."
      )
    if (
        self.bos_handling != BOSHandling.DO_NOTHING
        and self.bos_token_id is None
    ):
      raise ValueError(
          f"bos_token_id must be set if bos_handling is {self.bos_handling}."
      )
    if not set(self.meta_features).isdisjoint(set(self.bos_features)):
      raise ValueError(
          "bos_features and meta_features should not overlap. bos_features: "
          f"{self.bos_features}, meta_features: {self.meta_features}"
      )


_zeros = np.zeros


def _make_packed_buffer(length: int, x: np.ndarray | int) -> np.ndarray:
  """Defines the main buffers we will pack the data into."""
  x = np.asarray(x)
  shape = x.shape[1:]
  dtype = x.dtype
  return _zeros(
      shape=(length, *shape),
      dtype=dtype,
  )


def _make_packed_aux_info(length: int) -> np.ndarray:
  """Defines the auxiliar buffers for _segment_ids and _positions."""
  return _zeros(shape=(length,), dtype=np.int32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CtsElement:
  """Representation of a single element in the ConcatThenSplit implementation.

  Attributes:
    parent_state: The state of the parent iterator *before* __next__() was
      called.
    features: Features as returned by calling __next__() on the parent iterator.
    slices: If set then maps the feature name to the `slice` object for the
      split features.
  """

  parent_state: dict[str, Any]
  features: Mapping[str, Any]
  slices: Mapping[str, tuple[int, int]]

  def split(
      self, split_points: Mapping[str, int]
  ) -> tuple[_CtsElement | None, _CtsElement]:
    """Splits the element into two elements."""
    # We split at the very beginning.
    if all(x == 0 for x in split_points.values()):
      return None, self
    left_slices = {}
    right_slices = {}
    for key, p in split_points.items():
      sl = self.slices[key]
      if sl != _EMPTY_SLICE:
        start, stop = sl
      else:
        feature = self.features[key]
        start, stop = 0, (1 if np.ndim(feature) == 0 else len(feature))
      left_slices[key] = (start, p)
      right_slices[key] = (p, stop)
    left = dataclasses.replace(self, slices=left_slices)
    right = dataclasses.replace(self, slices=right_slices)
    return left, right

  def get_sliced_features(self, key: str) -> Mapping[str, Any]:
    feature = self.features[key]
    if np.ndim(feature) != 0 and key in self.slices:
      sl = self.slices[key]
      if sl != _EMPTY_SLICE:
        start, stop = self.slices[key]
        return feature[start:stop]
    return feature


@dataclasses.dataclass(kw_only=True)
class _CtsState:
  """Checkpoint state of the ConcatThenSplitDatasetIterator for PyGrain.

  Attributes:
    parent_state: The state of the parent iterator.
    remainder_slices: This parameter encodes both whether there is remainder
      element and (if yes) whether this element is sliced. The structure is a
      mapping str->slice. If all slices are empty then there is no remainder. If
      one or more slices are not empty, then there is a remainder element. The
      slices may contain the whole element or only an actual slice of it.
    elements_from_buffer_after_checkpoint: Number of elements from the buffer
      that were returned after the last checkpoint.
  """  # fmt: skip

  parent_state: dict[str, Any]
  remainder_slices: Mapping[str, tuple[int, int]]
  elements_from_buffer_after_checkpoint: int

  @property
  def has_remainder(self) -> bool:
    """Whether there is a remainder element."""
    for sl in self.remainder_slices.values():
      if sl != _EMPTY_SLICE:
        return True
    return False


class _ConcatThenSplitDatasetIterator(dataset.DatasetIterator):
  """Iterator for the concat-then-split packing transformation.

  See ConcatThenSplitIterDataset for details.

  Implementation details:
  It's important that we can efficiently restore the state and that our state
  is small and cheap to checkpoint. As a result we should not include actual
  data in the state. We can assume that the parent iterator state is cheap to
  copy and restore.
  We do this by only "checkpointing" when our state is simple - that is when
  there is no packed element in the buffer. See the docstring of _CtsState for
  details about the content of the state.
  Between "checkpoint" events we count the number of __next__() calls until the
  buffer is empty. The buffer will contain at most one packed element and we
  will have seen a number of elements. In practice this should be a small number
  of elements and a few steps should be fast.
  """

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      config: _CtsConfig,
  ):
    super().__init__(parent)
    self._config = config

    # Buffer for fully packed elements.
    self._packed_elements = collections.deque()
    # Potential element that is not yet packed.
    self._remainder_element: _CtsElement | None = None

    self._stop_iteration = False
    self._state = self._checkpoint()

    # Allocate and set a `np.range(max_position)` once. We will copy from this
    # into the *_positions features.
    self._arange = np.arange(
        max(self._config.length_struct.values()), dtype=np.int32
    )
    self._arange.flags.writeable = False

  def _has_full_length_feature(self, element: _CtsElement) -> bool:
    """Returns True if at least one features has its target sequence length."""
    for key, target_sequence_length in self._config.length_struct.items():
      feature = element.get_sliced_features(key)
      sequence_length = 1 if np.ndim(feature) == 0 else len(feature)
      if sequence_length < target_sequence_length:
        continue
      if sequence_length == target_sequence_length:
        return True
      if sequence_length > target_sequence_length:
        raise exceptions.PyGrainInternalError(
            f"Feature '{key}' has {sequence_length} tokens but target length is"
            f" only {target_sequence_length}. The element should be split."
        )
    return False

  def _pack_elements(
      self, elements: Sequence[_CtsElement]
  ) -> Mapping[str, Any]:
    """Packs the given elements into a single element."""
    if not elements:
      raise exceptions.PyGrainInternalError(f"No elements to pack in {self}.")
    packed_element = {}
    for key, sequence_length in self._config.length_struct.items():
      is_meta_feature = key in self._config.meta_features
      dummy_element = elements[0].features[key]

      values = _make_packed_buffer(sequence_length, dummy_element)
      if is_meta_feature:
        segment_ids = None
        positions = None
      else:
        segment_ids = _make_packed_aux_info(sequence_length)
        positions = _make_packed_aux_info(sequence_length)

      end = 0
      for segment_id, element in enumerate(elements, 1):
        # Insert `sequence` into `values[start:end]`.
        # For non-meta features also handle segment_ids and positions.
        sequence = element.get_sliced_features(key)
        sequence_length = 1 if np.ndim(sequence) == 0 else len(sequence)
        replace_with_bos = (
            self._config.bos_handling
            == BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS
            and key in self._config.bos_features
        )
        start = end
        end = start + sequence_length
        if end > len(values):
          raise exceptions.PyGrainInternalError(
              f"Feature {key!r}: Cannot pack sequence of length"
              f" {sequence_length} into segment [{start}:{end}] of values with"
              f" length {len(values)}. This is a bug in"
              " ConcatThenSplitPackDatasetIterator."
          )
        values[start:end] = sequence
        if replace_with_bos:
          values[start] = self._config.bos_token_id
        if not is_meta_feature:
          assert segment_ids is not None
          assert positions is not None
          segment_ids[start:end] = segment_id
          positions[start:end] = self._arange[: end - start]

      packed_element[key] = values
      if not is_meta_feature:
        packed_element[f"{key}_segment_ids"] = segment_ids
        packed_element[f"{key}_positions"] = positions

    return packed_element

  def _maybe_add_to_buffer(
      self,
      element: _CtsElement,
      *,
      buffer: list[_CtsElement],
      tokens_in_buffer: dict[str, int],
  ) -> _CtsElement | None:
    """Tries adding the element to the buffer.

    Args:
      element: The element to add to the buffer.
      buffer: The buffer to add the element to. Edited in place.
      tokens_in_buffer: Number of tokens per feature in the buffer. Edited in
        place.

    Returns:
      `None` if the element was fully added to the buffer without splitting.
      The unchanged element if it didn't fit into the buffer and couldn't be
        split.
      The remaining element if parts of the element were added to the buffer.
    """
    new_tokens_in_buffer: dict[str, int] = {}
    split_points: dict[str, int] = {}
    needs_splitting = False

    for key, target_sequence_length in self._config.length_struct.items():
      is_meta_feature = key in self._config.meta_features
      sequence = element.get_sliced_features(key)
      sequence_length = 1 if np.ndim(sequence) == 0 else len(sequence)
      new_tokens_in_buffer[key] = sequence_length
      assert target_sequence_length >= tokens_in_buffer[key]
      available_tokens = target_sequence_length - tokens_in_buffer[key]
      if is_meta_feature:
        if sequence_length > available_tokens:
          # Meta features cannot be split and must always fit.
          message = (
              f"Meta feature '{key}' limited packing. Resulting sequences"
              f" might have unnecessary padding. Feature '{key}' has"
              f" {sequence_length=} but only have"
              f" {available_tokens} available tokens"
              f" ({target_sequence_length=}). Consider increasing the"
              f" target sequence length for '{key}'."
          )
          logging.error(message)
          return element
        new_tokens_in_buffer[key] = sequence_length
      else:
        if sequence_length > available_tokens:
          needs_splitting = True
          split_points[key] = available_tokens
          new_tokens_in_buffer[key] = available_tokens
        else:
          # No splitting.
          split_points[key] = sequence_length
          new_tokens_in_buffer[key] = sequence_length

    if needs_splitting:
      element, remainder = element.split(split_points)
    else:
      remainder = None

    if element is not None:
      buffer.append(element)
    for k, v in new_tokens_in_buffer.items():
      tokens_in_buffer[k] += v
    return remainder

  @stats.record_next_duration_if_output
  def __next__(self):
    if self._packed_elements:
      self._state.elements_from_buffer_after_checkpoint += 1
      return self._stats.record_output_spec(self._packed_elements.popleft())
    if self._stop_iteration:
      raise StopIteration()

    # We have either nothing buffered or just the remainder element. This is a
    # good moment to update our state.
    self._state = self._checkpoint()

    # Buffer for elements to pack.
    buffer: list[_CtsElement] = []
    tokens_in_buffer = {k: 0 for k in self._config.length_struct.keys()}

    # Get elements from parent until the buffer has enough elements to fill
    # a packed element.
    # If elements from the parent are already considered packed, we add them
    # directly to `_packed_elements`.
    while True:
      if self._remainder_element is not None:
        self._remainder_element = self._maybe_add_to_buffer(
            self._remainder_element,
            buffer=buffer,
            tokens_in_buffer=tokens_in_buffer,
        )
        if self._remainder_element is not None:
          break

      parent_state = self._parent.get_state()
      try:
        features = next(self._parent)
      except StopIteration:
        self._stop_iteration = True
        break

      element_spec = tree_lib.spec_like(features)
      if not isinstance(element_spec, Mapping):
        raise ValueError(
            "Parent elements must be unnested dictionaries but got"
            f" {element_spec}."
        )
      if set(element_spec.keys()) != set(self._config.length_struct.keys()):
        raise ValueError(
            f"Parent element has structure {element_spec} but target sequence"
            f" length is {self._config.length_struct}. There must be a"
            " sequence length for each feature."
        )

      current_element = _CtsElement(
          parent_state=parent_state,
          features=features,
          slices=self._empty_slices(),
      )
      if not self._config.split_full_length_features:
        if self._has_full_length_feature(current_element):
          # The element has a full-length feature, so it's considered already
          # packed because split_full_length_features=False.
          self._packed_elements.append(self._pack_elements([current_element]))
          continue
      self._remainder_element = self._maybe_add_to_buffer(
          current_element, buffer=buffer, tokens_in_buffer=tokens_in_buffer
      )
      if self._remainder_element is not None:
        break

    if buffer:
      # The buffer is always the priority, so it's put at the left of the queue.
      self._packed_elements.appendleft(self._pack_elements(buffer))
    if self._stop_iteration:
      # Pack the remainder element if we have one.
      if self._remainder_element is not None:
        self._packed_elements.append(
            self._pack_elements([self._remainder_element])
        )
        self._remainder_element = None

    if self._packed_elements:
      self._state.elements_from_buffer_after_checkpoint += 1
      return self._stats.record_output_spec(self._packed_elements.popleft())

    # Buffers are empty. Good time to checkpoint again.
    self._state = self._checkpoint()

    if self._stop_iteration:
      raise StopIteration()
    raise exceptions.PyGrainInternalError(
        f"Failed to pack element from parent {self._parent} using config"
        f" {self._config}."
    )

  def _checkpoint(self) -> _CtsState:
    # Shouldn't have any elements buffered. Checkpointing that would be tricky.
    assert (
        not self._packed_elements
    ), f"Trying to checkpoint with non-empty buffer {self._packed_elements}"

    # Get status of remainder.
    if self._remainder_element is None:
      parent_state = self._parent.get_state()
      remainder_slices = self._empty_slices()
    else:
      parent_state = self._remainder_element.parent_state
      remainder_slices = self._remainder_element.slices

    return _CtsState(
        parent_state=parent_state,
        remainder_slices=remainder_slices,
        elements_from_buffer_after_checkpoint=0,
    )

  def _empty_slices(self) -> Mapping[str, tuple[int, int]]:
    """Empty slices when there is no remainder element."""
    return {
        feature: _EMPTY_SLICE
        for feature in self._config.length_struct
        if feature not in self._config.meta_features  # cannot be sliced
    }

  def get_state(self) -> dict[str, Any]:
    return dataclasses.asdict(self._state)

  def set_state(self, state: dict[str, Any]):
    state = _CtsState(**state)
    # Clear internal state.
    self._packed_elements = collections.deque()
    self._remainder_element = None
    self._stop_iteration = False

    self._parent.set_state(state.parent_state)
    has_remainder = state.has_remainder
    if not has_remainder:
      self._remainder_element = None
    else:
      try:
        features = next(self._parent)
      except StopIteration as e:
        raise ValueError(
            "Got unexpected StopIteration from parent when restoring"
            f" checkpoint {state} on {self}."
        ) from e
      self._remainder_element = _CtsElement(
          parent_state=state.parent_state,
          features=features,
          slices=state.remainder_slices,
      )

    self._state = self._checkpoint()
    # Advance until we have the same number of next calls.
    elements_from_buffer_after_checkpoint = (
        state.elements_from_buffer_after_checkpoint
    )
    timer = stats.Timer()
    with timer:
      for _ in range(elements_from_buffer_after_checkpoint):
        next(self)

    # Some warnings to see if our assumption about the state being easily
    # restored holds true in practice.
    if elements_from_buffer_after_checkpoint > 50:
      logging.warning(
          "[elements_from_buffer_after_checkpoint] While restoring the last"
          " checkpoint, we had to advance %d elements from the buffer.",
          elements_from_buffer_after_checkpoint,
      )
    duration = datetime.timedelta(microseconds=timer.value() / 1_000)
    if duration > datetime.timedelta(seconds=30):
      logging.warning(
          "[elements_from_buffer_after_checkpoint] Restoring the last"
          " checkpoint took more than 30 seconds: %s",
          duration,
      )


class ConcatThenSplitIterDataset(dataset.IterDataset):
  """Implements concat-then-split packing for sequence features.

  This assumes that elements of the parent dataset are unnested dictionaries
  and entries are either scalars or NumPy arrays. The first dimension is
  considered the sequence dimension and its size may vary between elements. All
  other dimensions must be the same size for all elements. Scalars are treated
  as 1-dimensional arrays of size 1.

  On a high level this concatenates the underlying dataset and then splits it
  at target sequence lengths intervals. This is well defined for the case of
  a single feature.
  For multiple features we start with an empty buffer and concatenate elements
  until at least one feature is fully packed.
  As an optimization, elements from the parent dataset that are already fully
  packed are passed through in priority.
  When the buffer contains enough elements to fill at least one feature to its
  target sequence length, we pack the buffer. The last element might not fully
  fit and will be split. The remainder of the split stays in the buffer.

  When packing features we also create {feature_name}_positions and
  {feature_name}_segment_ids features. They are 1D arrays of size
  sequence_length. Segment IDs start at 1 and enumerate the elements of the
  packed element. Positions indicate the position within the unpacked sequence.

  Features can be "meta features" in which case they are never split
  and we do not create *_positions and *_segment_ids features for them.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      length_struct: Mapping[str, int],
      meta_features: Collection[str] = (),
      split_full_length_features: bool = True,
      bos_handling: BOSHandling = BOSHandling.DO_NOTHING,
      bos_features: Collection[str] = (),
      bos_token_id: int | None = None,
  ):
    """Creates a dataset that concat-then-splits sequences from the parent.

    Args:
      parent: The parent dataset.
      length_struct: Mapping from feature name to target sequence length.
      meta_features: Set of feature names that are considered meta features.
        Meta features are never split and will be duplicated when other features
        of the same element are split. Otherwise, meta features are packed
        normally (they have their own sequence length). No *_positions and
        *_segment_ids features are created for meta features.
      split_full_length_features: Whether full-length features are split, or
        they are considered packed and passed through in priority. Setting
        split_full_length_features=False is an optimization when some sequences
        already have the target length, and you don't want them to be split.
        This optimization is not used by default.
      bos_handling: The instructions for handling BOS tokens (by default, no BOS
        token is added).
      bos_features: The features to which BOS handling is applied in case BOS is
        used.
      bos_token_id: The token indicating BOS in case BOS is used.
    """
    super().__init__(parent)
    self._config = _CtsConfig(
        length_struct=length_struct,
        meta_features=meta_features,
        split_full_length_features=split_full_length_features,
        bos_handling=bos_handling,
        bos_token_id=bos_token_id,
        bos_features=bos_features,
    )

  def __iter__(self) -> dataset.DatasetIterator:
    return _ConcatThenSplitDatasetIterator(
        self._parent.__iter__(), config=self._config
    )
