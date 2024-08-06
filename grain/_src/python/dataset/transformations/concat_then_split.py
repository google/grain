"""Provides ConcatThenSplitIterDataset.

This transformations takes elements with variable length sequence features,
concatenates (over the whole dataset) them
and then splits the result into sequence of the desired sequence length.
If elements contain a single sequence feature this will pack without padding.

If your elements shouldn't be splitted but need packing please use
FirstFitPackIterDataset.
"""

import collections
from collections.abc import Mapping, Sequence, Set
import dataclasses
import enum
from typing import Any

from grain._src.core import tree
from grain._src.python.dataset import dataset
import numpy as np


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConcatThenSplitConfig:
  """Config for ConcatThenSplitIterDataset.

  Attributes:
    sequence_lengths: Mapping from feature name to target sequence length.
    meta_features: Set of feature names that are considered meta features. Meta
      features are never splitted and will be duplicated when other features of
      the same element are splitted. Otherwise meta features are packed normally
      (they have their own sequence length). No *_postions and *_segment_ids
      features are created for meta features.
    insert_bos_after_split: If True insert BOS token after splitting an example.
      (doesn't apply to meta features).
    replace_with_bos_after_split: Deprecated. Only use for comparison with other
      ConcatThenSplit implementations. If True replace first token after
      splitting an example with BOS token.
    bos_id: Id of the BOS token.
  """

  sequence_lengths: Mapping[str, int]
  meta_features: Set[str] = frozenset()
  insert_bos_after_split: bool = False
  replace_with_bos_after_split: bool = False
  bos_id: int | None = None

  def __post_init__(self):
    if self.insert_bos_after_split and self.replace_with_bos_after_split:
      raise ValueError(
          "insert_bos_after_split and replace_with_bos_after_split cannot both "
          "be True at the same time."
      )
    if self.insert_bos_after_split and self.bos_id is None:
      raise ValueError("bos_id must be set if insert_bos_after_split is True.")
    if self.replace_with_bos_after_split and self.bos_id is None:
      raise ValueError(
          "bos_id must be set if replace_with_bos_after_split is True."
      )


def _zeros(*args, **kwargs) -> np.ndarray:
  return np.zeros(*args, **kwargs)


# Define the main buffers we will pack the data into.
def _make_packed_buffer(length: int, x: np.ndarray | int) -> np.ndarray:
  is_scalar = np.ndim(x) == 0
  if is_scalar:
    shape = ()
    dtype = np.int64 if isinstance(x, int) else np.asarray(x).dtype
  else:
    x = np.asarray(x)
    shape = x.shape[1:]
    dtype = x.dtype
  return _zeros(
      shape=(length, *shape),
      dtype=dtype,
  )


def _make_packed_aux_info(length: int) -> np.ndarray:
  return _zeros(shape=(length,), dtype=np.int32)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CtsElement:
  """Representation of a single element in the ConcatThenSplit implementation.

  Attributes:
    parent_state: The state of the parent iterator *before* __next__() was
      called.
    features: Features as returend by calling __next__() on the parent iterator.
    slices: If set then maps the feature name the `slice` object for the
      splitted features.
  """

  parent_state: dict[str, Any]
  features: Mapping[str, Any]
  slices: Mapping[str, slice] | None = None

  def split(
      self, split_points: Mapping[str, int]
  ) -> tuple["_CtsElement", "_CtsElement"]:
    """Splits the element into two elements."""
    left_slices = {}
    right_slices = {}
    for key, p in split_points.items():
      sl = self.slices.get(key) if self.slices else None
      if sl:
        start, stop = sl.start, sl.stop
      else:
        feature = self.features[key]
        start, stop = 0, (1 if np.ndim(feature) == 0 else len(feature))
      left_slices[key] = slice(start, p)
      right_slices[key] = slice(p, stop)
    left = dataclasses.replace(self, slices=left_slices)
    right = dataclasses.replace(self, slices=right_slices)
    return left, right

  def get_sliced_features(self, key: str) -> Mapping[str, Any]:
    feature = self.features[key]
    if np.ndim(feature) != 0 and self.slices and key in self.slices:
      return feature[self.slices[key]]
    return feature

  def get_offset(self, key: str) -> int:
    if self.slices:
      sl = self.slices.get(key)
      if sl:
        assert sl.start is not None
        return sl.start
    return 0


class _RemainderElementState(enum.Enum):
  """State of the remainder element."""

  NO_REMAINDER = 0
  REMAINDER_WITH_SLICE = 1
  REMAINDER_WITHOUT_SLICE = 2


class ConcatThenSplitDatasetIterator(dataset.DatasetIterator):
  """Iterator for the concat-then-split packing transformation.

  See ConcatThenSplitIterDataset for details.

  Implementation details:
  It's important that we can efficiently restore the state and that our state
  is small and cheap to checkpoint. As a result we should not include actual
  data in the state. We can assume that the parent iterator state is cheap to
  copy and restore.
  We do this by only "checkpointing" when our state is simple:
  - No packed elements in the buffer.
  - No remainder element or remainder element with known slice (after the last
    split)
  Between "checkpoint" events we count the number of __next__() calls. We assume
  that this will be a small number and we it's reasonable cheap to replay the
  last few steps or `set_state()`.
  """

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      config: ConcatThenSplitConfig,
  ):
    super().__init__()
    self._parent = parent
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
        max(self._config.sequence_lengths.values()), dtype=np.int32
    )
    self._arange.flags.writeable = False

  def _is_fully_packed(self, element: _CtsElement) -> bool:
    """Returns True at least one (non-meta) features has its sequence length."""
    for key, target_sequence_length in self._config.sequence_lengths.items():
      feature = element.get_sliced_features(key)
      sequence_length = 1 if np.ndim(feature) == 0 else len(feature)
      if sequence_length < target_sequence_length:
        continue
      if sequence_length == target_sequence_length:
        return True
      if sequence_length > target_sequence_length:
        raise ValueError(
            f"Feature {key} has {sequence_length} tokens but target length is "
            f" only {target_sequence_length}. Split the element."
        )
    return False

  def _pack_elements(
      self, elements: Sequence[_CtsElement]
  ) -> Mapping[str, Any]:
    """Pack the given elements into a single element."""
    if not elements:
      raise AssertionError(f"No elements to pack in {self}.")
    packed_element = {}
    # TODO: Could this be jitted with Numba?
    for key, sequence_length in self._config.sequence_lengths.items():
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
        if not is_meta_feature and element.get_offset(key) > 0:
          insert_bos = self._config.insert_bos_after_split
          replace_with_bos = self._config.replace_with_bos_after_split
        else:
          # Meta feature or not splitted.
          insert_bos = False
          replace_with_bos = False
        start = end
        end = start + sequence_length + int(insert_bos)
        if end > len(values):
          raise AssertionError(
              f"Feature {key!r}: Cannot pack sequence of length"
              f" {sequence_length} into segment [{start}:{end}] of values with"
              f" length {len(values)}. This is a bug in"
              " ConcatThenSplitPackDatasetIterator."
          )
        if insert_bos:
          values[start] = self._config.bos_id
          values[start + 1 : end] = sequence
        else:
          values[start:end] = sequence
        if replace_with_bos:
          values[start] = self._config.bos_id
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
      element: The element to add to the buffer. If the element was already
        splitted `is_splitted` should be set and the `features` should point to
        the remaining data.
      buffer:
      tokens_in_buffer:

    Returns:
      Returns `None` if the element was fully added to the buffer without
      splitting.
      Returns the unchanged element if it didn't fit into the buffer and
      couldn't be split.
      Returns the remaining element if parts of the element were added to the
      buffer.
    """
    new_tokens_in_buffer: dict[str, int] = {}
    split_points: dict[str, int] = {}
    needs_splitting = False

    for key, target_sequence_length in self._config.sequence_lengths.items():
      is_meta_feature = key in self._config.meta_features
      sequence = element.get_sliced_features(key)
      sequence_length = 1 if np.ndim(sequence) == 0 else len(sequence)
      insert_bos = False
      if not is_meta_feature and self._config.insert_bos_after_split:
        insert_bos = element.get_offset(key) > 0
      if insert_bos:
        sequence_length += 1
      new_tokens_in_buffer[key] = sequence_length
      assert target_sequence_length >= tokens_in_buffer[key]
      available_tokens = target_sequence_length - tokens_in_buffer[key]
      if is_meta_feature:
        if sequence_length > available_tokens:
          # Meta features cannot be split and must always fit.
          return element
        new_tokens_in_buffer[key] = sequence_length
      else:
        if insert_bos and available_tokens < 2:
          # No point in just inserting BOS.
          return element
        if sequence_length > available_tokens:
          needs_splitting = True
          split_points[key] = available_tokens
          new_tokens_in_buffer[key] = available_tokens
        else:
          # No splitting.
          split_points[key] = sequence_length
          new_tokens_in_buffer[key] = sequence_length

    if needs_splitting:
      if any(x > 0 for x in split_points.values()):
        element, remainder = element.split(split_points)
      else:
        element, remainder = None, element
    else:
      remainder = None

    if element is not None:
      buffer.append(element)
    for k, v in new_tokens_in_buffer.items():
      tokens_in_buffer[k] += v
    return remainder

  def __next__(self):
    if self._packed_elements:
      self._state["elements_from_buffer_after_checkpoint"] += 1
      return self._packed_elements.popleft()
    if self._stop_iteration:
      raise StopIteration

    # We have either nothing buffered or just the remainder element. This is a
    # good moment to update our state.
    self._state = self._checkpoint()

    # Buffer for elements to pack.
    buffer: list[_CtsElement] = []
    tokens_in_buffer = {k: 0 for k in self._config.sequence_lengths.keys()}

    # Get elements from parent until the buffer has enough elements to fill
    # a packed element.
    # If elements from the paraent are already considered packed we add them
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

      element_spec = tree.spec_like(features)
      if not isinstance(element_spec, Mapping):
        raise ValueError(
            "Parent elements must be unnested dictionaries but got"
            f" {element_spec}."
        )
      if set(element_spec.keys()) != set(self._config.sequence_lengths.keys()):
        raise ValueError(
            f"Parent element has structure {element_spec} but target sequence"
            f" length is {self._config.sequence_lengths}. There must be a"
            " sequence length for each feature."
        )

      current_element = _CtsElement(
          parent_state=parent_state,
          features=features,
      )
      if self._is_fully_packed(current_element):
        self._packed_elements.append(self._pack_elements([current_element]))
        continue
      self._remainder_element = self._maybe_add_to_buffer(
          current_element, buffer=buffer, tokens_in_buffer=tokens_in_buffer
      )
      if self._remainder_element is not None:
        break

    if buffer:
      self._packed_elements.append(self._pack_elements(buffer))
    if self._stop_iteration:
      # Pack the remainder element if we have one.
      if self._remainder_element is not None:
        self._packed_elements.append(
            self._pack_elements([self._remainder_element])
        )
        self._remainder_element = None

    if self._packed_elements:
      self._state["elements_from_buffer_after_checkpoint"] += 1
      return self._packed_elements.popleft()

    # Buffers are empty. Good time to checkpoint again.
    self._state = self._checkpoint()

    if self._stop_iteration:
      raise StopIteration
    raise AssertionError(
        f"Failed to pack element from parent {self._parent} using config "
        f"{self._config}. Please file a bug at https://github.com/google/grain/issues."
    )

  def _checkpoint(self) -> dict[str, Any]:
    # Shouldn't have any elements buffered. Checkpoint that gets tricky.
    assert not self._packed_elements

    # Get status of remainder.
    remainder_status = _RemainderElementState.NO_REMAINDER
    remainder_slices = None
    if self._remainder_element is None:
      parent_state = self._parent.get_state()
    else:
      parent_state = self._remainder_element.parent_state
      if self._remainder_element.slices is None:
        remainder_status = _RemainderElementState.REMAINDER_WITHOUT_SLICE
      else:
        remainder_status = _RemainderElementState.REMAINDER_WITH_SLICE
        remainder_slices = {
            k: [v.start, v.stop]
            for k, v in self._remainder_element.slices.items()
        }
    if not remainder_slices:
      # Put some dummy values to have consistent checkpoint shape.
      remainder_slices = {
          k: [-1, -1]
          for k in self._config.sequence_lengths.keys()
          if k not in self._config.meta_features
      }

    return {
        "parent_state": parent_state,
        "remainder_status": remainder_status.value,
        "remainder_slices": remainder_slices,
        "stop_iteration": self._stop_iteration,
        "elements_from_buffer_after_checkpoint": 0,
    }

  def get_state(self) -> dict[str, Any]:
    return self._state

  def set_state(self, state: dict[str, Any]):
    # Clear internal state.
    self._packed_elements = collections.deque()
    self._remainder_element = None
    self._stop_iteration = state["stop_iteration"]
    if self._stop_iteration:
      self._state = state
      return

    self._parent.set_state(state["parent_state"])
    remainder_status = _RemainderElementState(state["remainder_status"])
    if remainder_status == _RemainderElementState.NO_REMAINDER:
      self._remainder_element = None
    else:
      try:
        features = next(self._parent)
      except StopIteration as e:
        self._stop_iteration = True
        raise ValueError(
            "Got unexpected StopIteration from parent when restoring"
            f" checkpoint {state} on {self}."
        ) from e
      if remainder_status == _RemainderElementState.REMAINDER_WITH_SLICE:
        slices = state["remainder_slices"]
        slices = {k: slice(start, stop) for k, (start, stop) in slices.items()}
      else:
        slices = None
      self._remainder_element = _CtsElement(
          parent_state=state["parent_state"],
          features=features,
          slices=slices,
      )

    self._state = self._checkpoint()
    # Advance until we have the same number of next calls.
    for _ in range(state["elements_from_buffer_after_checkpoint"]):
      next(self)


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
  As an optimization elements from the parent dataset that are already fully
  packed are passed through.
  When the buffer contains enoguh elements to fill at least one feature to its
  target sequence length we pack the buffer. The last element might not fully
  fit and will be splitted. The remainder of the split stays in the buffer.

  When packing features we also create {feature_name}_positions and
  {feature_name}_segment_ids features
  that are 1D arrays of size sequence length. Segment IDs start at 1 and
  enumerate the elements of the packed element. Positions indicate the position
  within the unpacked sequence.

  Features can be "meta features" in which case they are never splitted
  and we do not create *_positions and *_segment_ids features for them.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      sequence_lengths: Mapping[str, int],
      meta_features: Set[str] | Sequence[str] = frozenset(),
      insert_bos_after_split: bool = False,
      replace_with_bos_after_split: bool = False,
      bos_id: int | None = None,
  ):
    """Creates a dataset that concat-then-splits sequences from the parent.

    Args:
      parent: The parent dataset.
      sequence_lengths: Mapping from feature name to target sequence length.
      meta_features: Set of feature names that are considered meta features.
        Meta features are never splitted and will be duplicated when other
        features of the same element are splitted. Otherwise meta features are
        packed normally (they have their own sequence length). No *_postions and
        *_segment_ids features are created for meta features.
      insert_bos_after_split: If True insert BOS token after splitting an
        example. (doesn't apply to meta features).
      replace_with_bos_after_split: Deprecated. Only use for comparison with
        other ConcatThenSplit implementations. If True replace first token after
        splitting an example with BOS token.
      bos_id: Id of the BOS token.
    """
    super().__init__(parent)
    self._config = ConcatThenSplitConfig(
        sequence_lengths=sequence_lengths,
        meta_features=set(meta_features),
        insert_bos_after_split=insert_bos_after_split,
        replace_with_bos_after_split=replace_with_bos_after_split,
        bos_id=bos_id,
    )

  def __iter__(self) -> dataset.DatasetIterator:
    return ConcatThenSplitDatasetIterator(
        self._parent.__iter__(), config=self._config
    )
