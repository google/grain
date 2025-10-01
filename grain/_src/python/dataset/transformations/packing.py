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
from typing import Any, Type

from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import packing_packed_batch
import numpy as np
import tree


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
      pack_alignment_struct: Any = None,
      padding_struct: Any = None,
      max_sequences_per_bin: int | None = None,
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
      pack_alignment_struct: Optional per-feature alignment values.
      padding_struct: Optional per-feature padding values.
      max_sequences_per_bin: Optional maximum number of input sequences that can
        be packed into a bin
    """
    super().__init__(parent)
    self._packer_cls = packer_cls
    self._num_packing_bins = num_packing_bins
    self._length_struct = length_struct
    self._seed = seed
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features
    self._pack_alignment_struct = pack_alignment_struct
    self._padding_struct = padding_struct
    self._max_sequences_per_bin = max_sequences_per_bin
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
    self._packed_batch_size_bytes = None

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

  def get_packed_batch_size_bytes(self) -> int:
    if self._packed_batch_size_bytes is None:
      raise ValueError(
          "No current batch. Size can only be computed after at least one"
          " element has been packed."
      )
    return self._packed_batch_size_bytes

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
          pack_alignment_struct=self._pack_alignment_struct,
          padding_struct=self._padding_struct,
          max_sequences_per_bin=self._max_sequences_per_bin,
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
          # Initialize the batch manager with the specific packer class: use
          # `element` to set dtypes + trailing dimensions. We are not adding the
          # element to the batch, just initializing it.
          self._current_batch = self._packer_cls(
              element,
              self._num_packing_bins,
              self._length_struct,
              meta_features=self._meta_features,
              pack_alignment_struct=self._pack_alignment_struct,
              padding_struct=self._padding_struct,
              max_sequences_per_bin=self._max_sequences_per_bin,
          )
          self._packed_batch_size_bytes = self._current_batch.get_size_bytes()

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
  """A generic dataset for packing transformations."""

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
      pack_alignment_struct: Any = None,
      padding_struct: Any = None,
      max_sequences_per_bin: int | None = None,
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
      pack_alignment_struct: Optional per-feature alignment values.
      padding_struct: Optional per-feature padding values.
      max_sequences_per_bin: Optional maximum number of input sequences that can
        be packed into a bin
    """
    super().__init__(parent)
    self._packer_cls = packer_cls
    self._length_struct = length_struct
    self._num_packing_bins = num_packing_bins
    self._seed = seed
    self._shuffle_bins = shuffle_bins
    self._shuffle_bins_group_by_feature = shuffle_bins_group_by_feature
    self._meta_features = meta_features
    self._pack_alignment_struct = pack_alignment_struct
    self._padding_struct = padding_struct
    self._max_sequences_per_bin = max_sequences_per_bin

    if max_sequences_per_bin is not None and max_sequences_per_bin <= 0:
      raise ValueError(
          "If specified, `max_sequences_per_bin` must be a positive integer,"
          f" but got {max_sequences_per_bin}."
      )

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
        pack_alignment_struct=self._pack_alignment_struct,
        padding_struct=self._padding_struct,
        max_sequences_per_bin=self._max_sequences_per_bin,
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
      pack_alignment_struct: Any = None,
      padding_struct: Any = None,
      max_sequences_per_bin: int | None = None,
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
      pack_alignment_struct: Optional per-feature alignment values.
      padding_struct: Optional per-feature padding values.
      max_sequences_per_bin: Optional maximum number of input sequences that can
        be packed into a bin
    """
    super().__init__(
        parent,
        packer_cls=packing_packed_batch.FirstFitPackedBatch,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        seed=seed,
        shuffle_bins=shuffle_bins,
        shuffle_bins_group_by_feature=shuffle_bins_group_by_feature,
        meta_features=meta_features,
        pack_alignment_struct=pack_alignment_struct,
        padding_struct=padding_struct,
        max_sequences_per_bin=max_sequences_per_bin,
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
      pack_alignment_struct: Any = None,
      padding_struct: Any = None,
      max_sequences_per_bin: int | None = None,
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
      pack_alignment_struct: Optional per-feature alignment values.
      padding_struct: Optional per-feature padding values.
      max_sequences_per_bin: Optional maximum number of input sequences that can
        be packed into a bin
    """
    super().__init__(
        parent,
        packer_cls=packing_packed_batch.BestFitPackedBatch,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        seed=seed,
        shuffle_bins=shuffle_bins,
        shuffle_bins_group_by_feature=shuffle_bins_group_by_feature,
        meta_features=meta_features,
        pack_alignment_struct=pack_alignment_struct,
        padding_struct=padding_struct,
        max_sequences_per_bin=max_sequences_per_bin,
    )

  def __str__(self) -> str:
    return "BestFitPackIterDataset"
