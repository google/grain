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

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any, Type

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import packing_packed_batch
import numpy as np
import tree


class _SegmentTree:
  """A segment tree data structure for efficiently finding the best-fitting bin."""

  def __init__(self, maxval: int):
    """Initializes the segment tree.

    Args:
      maxval: The maximum value that can be stored in the tree.
    """
    self.maxval = maxval
    # Tree size is 2 * maxval to accommodate leaves and internal nodes.
    self.tree = [0] * (2 * maxval)

  def add(self, val: int):
    """Adds a value to the tree."""
    assert 0 < val <= self.maxval
    i = self.maxval + val - 1
    self.tree[i] = val
    # Propagate the change up to the root.
    while i > 1:
      i >>= 1
      left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
      self.tree[i] = left if left >= right else right

  def remove(self, val: int):
    """Removes a value from the tree."""
    assert 0 < val <= self.maxval
    i = self.maxval + val - 1
    self.tree[i] = 0
    # Propagate the change up to the root.
    while i > 1:
      i >>= 1
      left, right = self.tree[i << 1], self.tree[(i << 1) + 1]
      self.tree[i] = left if left >= right else right

  def search(self, val: int) -> int:
    """Finds the smallest value in the tree that is >= val."""
    assert 0 < val <= self.maxval
    i = 1
    # Traverse down the tree to find the best fit.
    while i < self.maxval:
      # If the left child has a value large enough, go left.
      if self.tree[i << 1] >= val:
        i <<= 1
      # Otherwise, go right.
      else:
        i = (i << 1) + 1
    return self.tree[i]


class _BestFitPackedBatch(packing_packed_batch.PackedBatch):
  """Helper class that packs elements using the best-fit algorithm.

  This packer finds the bin with the smallest remaining space that can still
  accommodate the new element. This strategy aims to minimize padding by leaving
  larger contiguous blocks of space for future elements.
  """

  def __init__(
      self,
      element_for_shapes: Any,
      num_packing_bins: int,
      length_struct: Any,
      meta_features: Sequence[str] = (),
  ):
    """Initializes the BestFitPackedBatch.

    Args:
      element_for_shapes: An element to infer shapes and dtypes from.
      num_packing_bins: Number of bins to pack sequences into.
      length_struct: Target sequence length for each feature.
      meta_features: Meta features that do not need packing logic.
    """
    super().__init__(
        element_for_shapes,
        num_packing_bins,
        length_struct,
        meta_features=meta_features,
    )
    flat_lengths = tree.flatten(length_struct)
    if not any(l is not None for l in flat_lengths):
      raise ValueError("length_struct must contain at least one length.")
    self._max_length = max(l for l in flat_lengths if l is not None)

    # Use a segment tree to efficiently find the best-fitting bin.
    self._segment_tree = _SegmentTree(self._max_length)
    # Map remaining space to the list of bin indices with that space.
    self._space_to_bin_indices = defaultdict(deque)

    # Initialize all bins with the maximum available length.
    for i in range(num_packing_bins):
      self._segment_tree.add(self._max_length)
      self._space_to_bin_indices[self._max_length].append(i)

  def try_add_to_batch(self, element: Any) -> Any | None:
    """Tries to add an element to the batch using the best-fit strategy."""
    element_lengths = self._get_element_lengths(element)
    max_element_length = max(element_lengths.values())

    # Search for the smallest available space that fits the element.
    best_fit_space = self._segment_tree.search(max_element_length)

    # If no bin has enough space, return the failing components.
    if not best_fit_space or best_fit_space < max_element_length:
      return {k: v for k, v in element_lengths.items() if v > best_fit_space}

    # Get the index of the bin to use.
    bin_index = self._space_to_bin_indices[best_fit_space].popleft()

    # If this was the last bin with this specific amount of space, remove it.
    if not self._space_to_bin_indices[best_fit_space]:
      self._segment_tree.remove(best_fit_space)

    # Add the element to the selected bin.
    self._add_element_to_bin(element, bin_index, element_lengths)

    # Update the data structures with the new remaining space of the bin.
    new_space = self._remaining_lengths[bin_index]
    if new_space > 0:
      self._segment_tree.add(new_space)
      self._space_to_bin_indices[new_space].append(bin_index)

    return None


class PackingDatasetIterator(dataset.DatasetIterator):
  """A generic iterator for packing transformations.

  This iterator implements the core packing loop and state management but
  delegates the specific packing strategy (e.g., first-fit, best-fit) to a
  `packer_cls` provided at initialization. This allows for flexible packing
  algorithms while reusing the same iteration and batching logic.
  """

  def __init__(
      self,
      parent: dataset.DatasetIterator,
      *,
      packer_cls: Type[packing_packed_batch.PackedBatch],
      num_packing_bins: int,
      length_struct: Any,
      seed: int,
      shuffle_bins: bool,
      shuffle_bins_group_by_feature: str | None,
      meta_features: Sequence[str],
  ):
    """Initializes the generic packing iterator.

    Args:
      parent: The parent iterator.
      packer_cls: The class responsible for the packing logic. Must be a
        subclass of `packing_packed_batch.PackedBatch`.
      num_packing_bins: Number of bins to pack sequences into.
      length_struct: Target sequence length for each feature.
      seed: Random seed for shuffling.
      shuffle_bins: Whether to shuffle bins after packing.
      shuffle_bins_group_by_feature: Feature to group by for shuffling.
      meta_features: Meta features that do not require packing.
    """
    super().__init__(parent)
    self._packer_cls = packer_cls
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._seed = seed
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features
    self._reset()

  def _reset(self):
    self._current_batch = None  # Not yet fully packed.
    # The parent state to restore to get back to our current state.
    self._last_emitted_batch_parent_state = self._parent.get_state()
    # The parent state after constructing the current completed packed batch.
    self._current_batch_parent_state = None
    # If available, fully packed but rows [:self._next_row] were already
    # emitted.
    self._packed_batch = None
    # The last packed batch can be partial.
    self._packed_batch_num_bins = None
    # _next_row gets reset between batches.
    self._next_row = 0
    # _counter is a global counter for rows emitted.
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
    seed = self._seed + self._counter - self._next_row
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
      # Use the provided packer class to create the new batch manager.
      self._current_batch = self._packer_cls(
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
          # Finalize and yield the last partial batch.
          with timer:
            self._finalize_current_batch(None)
            self._current_batch_parent_state = prior_iterator_state
          return next(self)
        else:
          # The inner iterator is exhausted and there is no current batch.
          raise StopIteration() from e

      with timer:
        # Remove elements not in packing struct.
        element = tree.map_structure_up_to(
            self._length_struct, lambda x: x, element
        )

        if self._current_batch is None:
          # Initialize the batch manager with the specific packer class.
          self._current_batch = self._packer_cls(
              element,
              self._num_packing_bins,
              self._length_struct,
              meta_features=self._meta_features,
          )

        # Try adding element to the current packed batch.
        failing_components = self._current_batch.try_add_to_batch(element)

      # When the batch is full, yield the packed data and start a new batch.
      if failing_components is not None:
        with timer:
          self._finalize_current_batch(element)
          self._current_batch_parent_state = prior_iterator_state
          assert self._current_batch is not None

          if self._current_batch.try_add_to_batch(element) is not None:
            # If a single example can't fit in an empty batch, it's an error.
            element_shape = tree.map_structure(lambda x: x.shape, element)
            raise ValueError(
                "Could not add element to empty packed batch! Packed batch has"
                f" packing sequence_lengths: {self._length_struct} while"
                f" element has shape: {element_shape}"
            )
        # We now have a packed batch.
        return next(self)


class PackIterDataset(dataset.IterDataset):
  """A generic dataset for packing transformations.

  This dataset acts as a factory for `PackingDatasetIterator`, allowing different
  packing strategies to be plugged in via the `packer_cls` argument.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      packer_cls: Type[packing_packed_batch.PackedBatch],
      length_struct: Any,
      num_packing_bins: int,
      seed: int = 0,
      shuffle_bins: bool = True,
      shuffle_bins_group_by_feature: str | None = None,
      meta_features: Sequence[str] = (),
  ):
    """Initializes the generic packing dataset.

    Args:
      parent: Parent dataset with variable length sequences.
      packer_cls: The class implementing the packing algorithm.
      length_struct: Target sequence length for each feature.
      num_packing_bins: Number of bins to pack sequences into.
      seed: Random seed for shuffling bins.
      shuffle_bins: Whether to shuffle bins after packing.
      shuffle_bins_group_by_feature: Feature to group by for shuffling.
      meta_features: Meta features that do not need packing logic.
    """
    super().__init__(parent)
    self._packer_cls = packer_cls
    self._length_struct = length_struct
    self._num_packing_bins = num_packing_bins
    self._seed = seed
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features

  def __iter__(self) -> dataset.DatasetIterator:
    return PackingDatasetIterator(
        self._parent.__iter__(),
        packer_cls=self._packer_cls,
        num_packing_bins=self._num_packing_bins,
        length_struct=self._length_struct,
        seed=self._seed,
        shuffle_bins=self._shuffle_bins,
        shuffle_bins_group_by_feature=self._shuffle_bins_group_by_feature,
        meta_features=self._meta_features,
    )


class FirstFitPackIterDataset(PackIterDataset):
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
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      length_struct: Any,
      num_packing_bins: int,
      seed: int = 0,
      shuffle_bins: bool = True,
      shuffle_bins_group_by_feature: str | None = None,
      meta_features: Sequence[str] = (),
  ):
    """Creates a dataset that packs sequences using the first-fit strategy.

    Args:
      parent: Parent dataset with variable length sequences.
      length_struct: Target sequence length for each feature.
      num_packing_bins: Number of bins to pack sequences into.
      seed: Random seed for shuffling bins.
      shuffle_bins: Whether to shuffle bins after packing.
      shuffle_bins_group_by_feature: Feature to group by for shuffling.
      meta_features: Meta features that do not need packing logic.
    """
    super().__init__(
        parent,
        # Provide the specific packer class for the First-Fit strategy.
        packer_cls=packing_packed_batch.PackedBatch,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        seed=seed,
        shuffle_bins=shuffle_bins,
        shuffle_bins_group_by_feature=shuffle_bins_group_by_feature,
        meta_features=meta_features,
    )

  def __str__(self) -> str:
    return "FirstFitPackIterDataset"


class BestFitPackIterDataset(PackIterDataset):
  """Implements best-fit packing of sequences.

  The best-fit algorithm attempts to pack elements more efficiently than
  first-fit by placing each new element into the bin that will leave the
  smallest remaining space (i.e., the "tightest" fit). This can lead to less
  overall padding compared to the simpler first-fit approach, especially when
  element sizes vary significantly.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      *,
      length_struct: Any,
      num_packing_bins: int,
      seed: int = 0,
      shuffle_bins: bool = True,
      shuffle_bins_group_by_feature: str | None = None,
      meta_features: Sequence[str] = (),
  ):
    """Creates a dataset that packs sequences using the best-fit strategy.

    Args:
      parent: Parent dataset with variable length sequences.
      length_struct: Target sequence length for each feature.
      num_packing_bins: Number of bins to pack sequences into.
      seed: Random seed for shuffling bins.
      shuffle_bins: Whether to shuffle bins after packing.
      shuffle_bins_group_by_feature: Feature to group by for shuffling.
      meta_features: Meta features that do not need packing logic.
    """
    super().__init__(
        parent,
        # Provide the specific packer class for the Best-Fit strategy.
        packer_cls=_BestFitPackedBatch,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        seed=seed,
        shuffle_bins=shuffle_bins,
        shuffle_bins_group_by_feature=shuffle_bins_group_by_feature,
        meta_features=meta_features,
    )

  def __str__(self) -> str:
    return "BestFitPackIterDataset"
