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

from collections.abc import Sequence
from typing import Any

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import packing_packed_batch
import numpy as np
import tree


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
      shuffle_bins_group_by_feature: str | None = None,
      meta_features: Sequence[str] = (),
  ):
    """Creates a dataset that packs sequences from the parent dataset.

    Args:
      parent: Parent dataset with variable length sequences. Sequence cannot be
        longer than their length_struct value.
      length_struct: Target sequence length for each feature.
      num_packing_bins: Number of bins to pack sequences into.
      shuffle_bins: Whether to shuffle bins after packing.
      shuffle_bins_group_by_feature: No-op if shuffle_bins is False. When
        shuffle_bins is True, if shuffle_bins_group_by_feature is set to
        something non-None, we will group the bins by this feature name and
        shuffle within each group. If None, the entire batch is shuffled without
        regard to this feature. The primary use case for this is to only shuffle
        within each epoch to avoid epoch leakage.
      meta_features: Meta features that do not need *_segment_ids and
        *_positions features.
    """
    super().__init__(parent)
    self._length_struct = length_struct
    self._num_packing_bins = num_packing_bins
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features

  def __str__(self) -> str:
    return "FirstFitPackIterDataset"

  def __iter__(self) -> dataset.DatasetIterator:
    return FirstFitPackDatasetIterator(
        self._parent.__iter__(),
        num_packing_bins=self._num_packing_bins,
        length_struct=self._length_struct,
        shuffle_bins=self._shuffle_bins,
        shuffle_bins_group_by_feature=self._shuffle_bins_group_by_feature,
        meta_features=self._meta_features,
    )


class FirstFitPackDatasetIterator(dataset.DatasetIterator):
  """Iterator for the first-fit packing transformation."""

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      num_packing_bins: int,
      length_struct: Any,  # PyTree[int | None],
      shuffle_bins: bool,
      shuffle_bins_group_by_feature: str | None,
      meta_features: Sequence[str],
  ):
    super().__init__(parent)
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features
    self._reset()

  def _reset(self):
    self._current_batch = None  # Not yet fully packed.
    # The parent state to restore to get back to our current state, i.e. the
    # state that the parent was in after producing all elements used in the last
    # fully emitted batch. We only advance this when we finish emitting a packed
    # batch.
    self._last_emitted_batch_parent_state = self._parent.get_state()
    # The parent state after constructing the current completed packed batch,
    # right before adding the element that failed to fit.
    self._current_batch_parent_state = None
    # If available, fully packed but rows [:self._next_row] were already
    # emitted.
    self._packed_batch = None
    # The last packed batch can be partial and have few bins with elements.
    self._packed_batch_num_bins = None
    # _next_row gets reset between batches.
    # _counter is a global counter for rows emitted, does not get reset.
    self._next_row = 0
    self._counter = 0  # Used for RNG seed.
    self._shuffled_rows = None

  def get_state(self) -> dict[str, Any]:
    return {
        "parent": self._last_emitted_batch_parent_state,
        "next_row": self._next_row,
        "counter": self._counter,
    }

  def set_state(self, state: dict[str, Any]):
    self._parent.set_state(state["parent"])
    self._reset()
    self._next_row = state["next_row"]
    self._counter = state["counter"]

  def _generate_and_set_shuffled_rows(self):
    assert self._packed_batch_num_bins is not None
    seed = self._counter - self._next_row
    self._shuffled_rows = np.random.default_rng(seed).permuted(
        range(self._packed_batch_num_bins)
    )
    if self._shuffle_bins_group_by_feature is not None:
      unique_groups_in_row = [
          np.unique(row)
          for row in self._packed_batch[self._shuffle_bins_group_by_feature]
      ]
      average_group_per_row = [
          # nan_to_num is for the divide by zero case.
          np.nan_to_num(np.sum(s) / np.count_nonzero(s), nan=0.0)
          for s in unique_groups_in_row
      ]
      tuples_to_sort = [
          (average_group_per_row[i], self._shuffled_rows[i], i)
          for i in range(self._packed_batch_num_bins)
      ]
      # Sort by average group followed by original shuffle position.
      self._shuffled_rows = [t[2] for t in sorted(tuples_to_sort)]

  def _finalize_current_batch(self, element_for_shapes):
    assert self._current_batch is not None
    self._packed_batch = self._current_batch.get_packed_batch()
    # Detect number of bins. The last batch can be partial.
    self._packed_batch_num_bins = max(
        tree.flatten(
            tree.map_structure(lambda x: x.shape[0], self._packed_batch)
        )
    )
    assert self._packed_batch_num_bins <= self._num_packing_bins
    if self._shuffle_bins:
      self._generate_and_set_shuffled_rows()

    if element_for_shapes is None:
      self._current_batch = None
    else:
      self._current_batch = packing_packed_batch.PackedBatch(
          element_for_shapes,
          self._num_packing_bins,
          self._length_struct,
          meta_features=self._meta_features,
      )

  @dataset_stats.record_next_duration_if_output
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
          self._last_emitted_batch_parent_state = (
              self._current_batch_parent_state
          )
          self._current_batch_parent_state = None
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

  def __str__(self) -> str:
    return "FirstFitPackDatasetIterator"
