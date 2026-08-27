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
"""Tests for slice + shuffle + repeat interaction.

These tests verify that slicing a dataset to a subset, shuffling, and repeating
produces the correct repeated-and-shuffled subset when using wrap_around=True.

cl/805818068 changed SliceMapDataset's index computation to propagate epoch
information through the parent length, which breaks when shuffle sits between
slice and repeat (causing double epoch-counting). The wrap_around=True parameter
restores the old modular wrapping behavior for this case.
"""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset


class SliceShuffleRepeatWrapAroundTest(parameterized.TestCase):
  """Tests that wrap_around=True fixes slice+shuffle+repeat interaction."""

  def test_slice_shuffle_repeat_stays_within_subset(self):
    """Slicing with wrap_around then shuffling then repeating stays in subset."""
    parent_size = 100
    subset_size = 5
    num_to_read = 20  # 4 epochs worth

    ds = dataset.MapDataset.range(parent_size)
    ds = ds.slice(slice(subset_size), wrap_around=True)
    ds = ds.shuffle(seed=42)
    ds = ds.repeat()

    values = list(itertools.islice(ds, num_to_read))

    # All values should come from the original subset [0, 4].
    for v in values:
      self.assertGreaterEqual(int(v), 0)
      self.assertLess(int(v), subset_size)

    # We should see all values from the subset represented.
    self.assertEqual(set(int(v) for v in values), set(range(subset_size)))

  def test_slice_shuffle_repeat_values_match_unsliced(self):
    """ds[:K] with wrap_around + shuffle + repeat matches ds.shuffle.repeat when K == len(ds)."""
    num_to_read = 30

    ds = dataset.MapDataset.range(10)
    ds_sliced = (
        ds.slice(slice(len(ds)), wrap_around=True).shuffle(seed=42).repeat()
    )
    ds_unsliced = ds.shuffle(seed=42).repeat()

    sliced_vals = [int(v) for v in itertools.islice(ds_sliced, num_to_read)]
    unsliced_vals = [int(v) for v in itertools.islice(ds_unsliced, num_to_read)]

    self.assertSequenceEqual(sliced_vals, unsliced_vals)

  @parameterized.parameters(
      dict(parent_size=1000, subset_size=4, seed=42),
      dict(parent_size=50, subset_size=10, seed=0),
      dict(parent_size=200, subset_size=1, seed=99),
  )
  def test_slice_shuffle_repeat_subset_containment(
      self, parent_size, subset_size, seed
  ):
    """Values from slice(wrap_around)+shuffle+repeat must come from subset."""
    num_to_read = subset_size * 5  # Read 5 epochs

    ds = dataset.MapDataset.range(parent_size)
    ds = ds.slice(slice(subset_size), wrap_around=True)
    ds = ds.shuffle(seed=seed)
    ds = ds.repeat()

    values = [int(v) for v in itertools.islice(ds, num_to_read)]

    for v in values:
      self.assertIn(
          v,
          set(range(subset_size)),
          f'Value {v} is outside the subset [0, {subset_size}). '
          'This indicates the slice+shuffle+repeat pipeline is reading '
          'beyond the intended subset.',
      )

  def test_slice_with_offset_shuffle_repeat(self):
    """Slicing with offset + wrap_around + shuffle + repeat stays in subset."""
    parent_size = 100
    offset = 10
    subset_size = 5
    num_to_read = 20

    ds = dataset.MapDataset.range(parent_size)
    ds = ds.slice(slice(offset, offset + subset_size), wrap_around=True)
    ds = ds.shuffle(seed=42)
    ds = ds.repeat()

    values = [int(v) for v in itertools.islice(ds, num_to_read)]
    expected_subset = set(range(offset, offset + subset_size))

    for v in values:
      self.assertIn(
          v,
          expected_subset,
          f'Value {v} is outside the subset {expected_subset}.',
      )

  def test_slice_repeat_without_shuffle_preserves_values(self):
    """Slice(wrap_around) + repeat (no shuffle) cycles through the subset."""
    parent_size = 100
    subset_size = 3
    num_to_read = 9

    ds = dataset.MapDataset.range(parent_size)
    ds = ds.slice(slice(subset_size), wrap_around=True)
    ds = ds.repeat()

    values = [int(v) for v in itertools.islice(ds, num_to_read)]
    # Should be [0, 1, 2, 0, 1, 2, 0, 1, 2]
    self.assertSequenceEqual(values, [0, 1, 2] * 3)

  def test_mix_slice_repeat_stays_within_subset(self):
    """Mixing datasets, slicing with wrap_around, and repeating stays within."""
    a = dataset.MapDataset.range(5)
    b = dataset.MapDataset.range(10, 15)

    ds = dataset.MapDataset.mix([a, b], weights=[0.7, 0.3])
    subset_size = 3
    ds_subset = ds.slice(slice(subset_size), wrap_around=True)

    # Get the actual first 3 elements of the mixed dataset.
    expected_values = set(
        int(ds_subset[i])  # pyrefly: ignore[bad-argument-type]
        for i in range(subset_size)
    )

    ds_repeated = ds_subset.repeat()
    values = [int(v) for v in itertools.islice(ds_repeated, 12)]

    for v in values:
      self.assertIn(
          v,
          expected_values,
          f'Value {v} is not in the expected subset {expected_values}.',
      )


class SliceMapWithIndexMultiEpochTest(parameterized.TestCase):
  """Tests that default (non-wrap_around) behavior preserves cl/805818068 fix."""

  def test_slicing_with_index_multi_epoch_default(self):
    """Default behavior: ds[:K].repeat() matches ds.repeat() for map_with_index."""
    num_to_compare = 20

    ds = dataset.MapDataset.range(10)
    ds = ds.map_with_index(lambda i, x: {'index': i, 'value': x})

    ds_sliced = ds[: len(ds)]
    ds_sliced = ds_sliced.repeat()
    ds_sliced = list(itertools.islice(ds_sliced, num_to_compare))

    ds_unsliced = ds.repeat()
    ds_unsliced = list(itertools.islice(ds_unsliced, num_to_compare))

    self.assertSequenceEqual(ds_sliced, ds_unsliced)


if __name__ == '__main__':
  absltest.main()
