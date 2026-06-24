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
"""Tests for slice transformation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
import grain._src.python.dataset.transformations.slice as slice_ds
import numpy as np
from typing_extensions import override


class EmptyMapDataset(dataset.MapDataset[int]):

  def __init__(self):
    super().__init__(parents=[])

  @override
  def __len__(self) -> int:
    return 0

  @override
  def __getitem__(self, index):
    raise IndexError("Index out of range")


class SliceMapDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 1, 20),
      (0, 2, 10),
      (1, 2, 10),
      (0, 3, 7),
      (1, 3, 7),
      (2, 3, 6),
      (30, 100, 0),
  )
  def test_len(self, start: int, step: int, expected_len: int):
    ds = dataset.MapDataset.range(20)
    sl = slice(start, 20, step)
    range_ds_for_process = slice_ds.SliceMapDataset(ds, sl)
    self.assertLen(range_ds_for_process, expected_len)

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_getitem(self, start: int, stop: int, step: int):
    ds = dataset.MapDataset.range(20)
    ds = slice_ds.SliceMapDataset(ds, slice(start, stop, step))
    ds_items = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(ds_items, list(range(20))[start:stop:step])

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_getitem_slice(self, start: int, stop: int, step: int):
    ds = dataset.MapDataset.range(20)
    ds = ds[start:stop:step]
    ds_items = [ds[i] for i in range(len(ds))]
    self.assertSequenceEqual(ds_items, list(range(20))[start:stop:step])

  def test_getitem_with_slice_on_sliced_dataset(self):
    ds = dataset.MapDataset.range(20)
    # First slice creates a SliceMapDataset
    sliced_ds = ds[2:18:2]  # Expected: [2, 4, 6, 8, 10, 12, 14, 16]
    self.assertIsInstance(sliced_ds, slice_ds.SliceMapDataset)
    self.assertSequenceEqual(list(sliced_ds), [2, 4, 6, 8, 10, 12, 14, 16])

    # Second slice on the already sliced dataset
    re_sliced_ds = sliced_ds[1:6:2]  # Expected: [4, 8, 12]
    self.assertIsInstance(re_sliced_ds, slice_ds.SliceMapDataset)
    self.assertSequenceEqual(list(re_sliced_ds), [4, 8, 12])

  @parameterized.parameters(
      itertools.product(range(-8, 8), range(-9, 8), [-2, -1, 1, 2])
  )
  def test_iter(self, start: int, stop: int, step: int):
    ds = dataset.MapDataset.range(20)
    ds = slice_ds.SliceMapDataset(ds, slice(start, stop, step))
    ds_iter = iter(ds)
    ds_items = list(ds_iter)
    self.assertSequenceEqual(ds_items, list(range(20))[start:stop:step])

  @parameterized.parameters(
      dict(
          slice_args=(1, 8, 2),  # range(20)[1:8:2] -> [1, 3, 5, 7]
          indices=[0, 1, 3],
          expected_data=[1, 3, 7],
      ),
      dict(
          slice_args=(None, 5, None),  # range(20)[:5] -> [0, 1, 2, 3, 4]
          indices=[0, 1, 2, 3, 4],
          expected_data=[0, 1, 2, 3, 4],
      ),
      dict(
          slice_args=(8, 2, -2),  # range(20)[8:2:-2] -> [8, 6, 4]
          indices=[0, 2],
          expected_data=[8, 4],
      ),
      dict(
          slice_args=(
              5,
              None,
              None,
          ),  # range(20)[5:] -> [5, 6, ..., 19], len 15
          indices=[15, 16],  # next epoch
          expected_data=[5, 6],
      ),
  )
  def test_getitems(self, slice_args, indices, expected_data):
    ds = dataset.MapDataset.range(20)
    ds = slice_ds.SliceMapDataset(ds, slice(*slice_args))
    actual_data = ds._getitems(indices)
    self.assertSequenceEqual(actual_data, expected_data)

  def test_slice_of_empty_dataset_is_empty(self):
    ds = EmptyMapDataset()
    ds = slice_ds.SliceMapDataset(ds, slice(0, 10))
    self.assertEmpty(ds)

  def test_init_raises_error_for_non_slice_object(self):
    ds = dataset.MapDataset.range(10)
    with self.assertRaisesRegex(ValueError, "sl is not a slice: <class 'int'>"):
      slice_ds.SliceMapDataset(ds, 5)  # type: ignore

  def test_accessing_items_beyond_len_minus_one_succeeds(self):
    ds = dataset.MapDataset.range(20)
    ds = slice_ds.SliceMapDataset(ds, slice(5))  # 0, 1, 2, 3, 4
    self.assertLen(ds, 5)
    self.assertEqual(ds[5], 0)
    self.assertEqual(ds[13], 3)
    self.assertEqual(ds[42], 2)

  def test_composing_slices_contains_correct_elements(self):
    ds = dataset.MapDataset.range(20)
    ds = slice_ds.SliceMapDataset(ds, slice(0, 15, 3))  # 0, 3, 6, 9, 12
    ds = slice_ds.SliceMapDataset(ds, slice(0, 20, 2))  # 0, 6, 12
    self.assertSequenceEqual(list(ds), [0, 6, 12])

  def test_slicing_with_index_multi_epoch(self):
    num_to_compare = 20

    ds = dataset.MapDataset.range(10)
    ds = ds.map_with_index(lambda i, x: {"index": i, "value": x})

    ds_sliced = ds[: len(ds)]
    ds_sliced = ds_sliced.repeat()
    ds_sliced = list(itertools.islice(ds_sliced, num_to_compare))

    ds_unsliced = ds.repeat()
    ds_unsliced = list(itertools.islice(ds_unsliced, num_to_compare))

    self.assertSequenceEqual(ds_sliced, ds_unsliced)

  def test_element_spec(self):
    ds = dataset.MapDataset.range(2)
    ds = slice_ds.SliceMapDataset(ds, slice(0, 1))
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())


class ReindexMapDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(indices=[3, 1, 4, 1, 5], expected=[3, 1, 4, 1, 5]),
      dict(indices=[0, 1, 2, 3, 4], expected=[0, 1, 2, 3, 4]),
      dict(indices=[4, 3, 2, 1, 0], expected=[4, 3, 2, 1, 0]),
      dict(indices=[0, 0, 0, 0], expected=[0, 0, 0, 0]),
      dict(indices=[7], expected=[7]),
      dict(indices=[0, 1, 2], expected=[0, 1, 2]),
      dict(indices=[9, 8, 7], expected=[9, 8, 7]),
      dict(indices=[5], expected=[5]),
  )
  def test_reindex(self, indices, expected):
    ds = dataset.MapDataset.range(10)
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    self.assertLen(reindexed, len(expected))
    self.assertSequenceEqual(list(reindexed), expected)

  def test_reindex_with_map_dataset_indices(self):
    parent = dataset.MapDataset.range(10)
    indices = dataset.MapDataset.range(5)  # [0, 1, 2, 3, 4]
    reindexed = slice_ds.ReindexMapDataset(parent, indices)
    self.assertLen(reindexed, 5)
    self.assertSequenceEqual(list(reindexed), [0, 1, 2, 3, 4])

  def test_reindex_with_map_dataset_indices_sliced(self):
    parent = dataset.MapDataset.range(20)
    # Indices are [0, 2, 4, 6, 8] from a sliced MapDataset.
    indices = dataset.MapDataset.range(10)[::2]
    reindexed = slice_ds.ReindexMapDataset(parent, indices)
    self.assertLen(reindexed, 5)
    self.assertSequenceEqual(list(reindexed), [0, 2, 4, 6, 8])

  def test_multi_epoch_access(self):
    """Accessing beyond len(ds) wraps indices and advances parent epoch."""
    parent = dataset.MapDataset.range(10)
    indices = [3, 7]
    reindexed = slice_ds.ReindexMapDataset(parent, indices)
    self.assertLen(reindexed, 2)
    # First epoch: indices [3, 7] -> parent values [3, 7].
    self.assertEqual(reindexed[0], 3)
    self.assertEqual(reindexed[1], 7)
    # Second epoch: indices [3+10, 7+10] -> parent wraps -> values [3, 7].
    self.assertEqual(reindexed[2], 3)
    self.assertEqual(reindexed[3], 7)

  def test_multi_epoch_with_map_with_index(self):
    """Verify multi-epoch indices are passed correctly through the pipeline."""
    parent = dataset.MapDataset.range(10)
    parent = parent.map_with_index(lambda i, x: {"index": i, "value": x})
    indices = [2, 5]
    reindexed = slice_ds.ReindexMapDataset(parent, indices)
    # First epoch.
    self.assertEqual(reindexed[0], {"index": 2, "value": 2})
    self.assertEqual(reindexed[1], {"index": 5, "value": 5})
    # Second epoch: parent index advances by parent_length.
    self.assertEqual(reindexed[2], {"index": 12, "value": 2})
    self.assertEqual(reindexed[3], {"index": 15, "value": 5})

  def test_getitems(self):
    ds = dataset.MapDataset.range(10)
    indices = [3, 1, 4, 1, 5]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    actual = reindexed._getitems([0, 2, 4])
    self.assertSequenceEqual(actual, [3, 4, 5])

  def test_getitems_multi_epoch(self):
    ds = dataset.MapDataset.range(10)
    indices = [3, 7]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    # Index 2 and 3 are in the second epoch.
    actual = reindexed._getitems([0, 1, 2, 3])
    self.assertSequenceEqual(actual, [3, 7, 3, 7])

  def test_slice_on_reindexed_dataset(self):
    """Slicing a ReindexMapDataset should produce a SliceMapDataset."""
    ds = dataset.MapDataset.range(10)
    indices = [3, 1, 4, 1, 5]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    sliced = reindexed[1:4]
    self.assertIsInstance(sliced, slice_ds.SliceMapDataset)
    self.assertSequenceEqual(list(sliced), [1, 4, 1])

  def test_iter(self):
    ds = dataset.MapDataset.range(10)
    indices = [9, 0, 5]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    self.assertSequenceEqual(list(iter(reindexed)), [9, 0, 5])

  def test_str_short_indices(self):
    ds = dataset.MapDataset.range(10)
    indices = [1, 2, 3]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    self.assertEqual(
        str(reindexed), "ReindexMapDataset(indices=[1, 2, 3], len=3)"
    )

  def test_str_long_indices(self):
    ds = dataset.MapDataset.range(100)
    indices = list(range(10))
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    self.assertEqual(
        str(reindexed),
        "ReindexMapDataset(indices=[0, 1, ..., 8, 9], len=10)",
    )

  def test_element_spec(self):
    ds = dataset.MapDataset.range(10)
    indices = [1, 2, 3]
    reindexed = slice_ds.ReindexMapDataset(ds, indices)
    spec = dataset.get_element_spec(reindexed)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())

  def test_via_map_dataset_slice_with_list(self):
    """Test that MapDataset.slice dispatches to ReindexMapDataset for lists."""
    ds = dataset.MapDataset.range(10)
    reindexed = ds.slice([3, 1, 4])
    self.assertIsInstance(reindexed, slice_ds.ReindexMapDataset)
    self.assertSequenceEqual(list(reindexed), [3, 1, 4])

  def test_via_map_dataset_slice_with_map_dataset(self):
    """Test that MapDataset.slice works with a MapDataset of indices."""
    parent = dataset.MapDataset.range(10)
    indices = dataset.MapDataset.range(3)
    reindexed = parent.slice(indices)
    self.assertIsInstance(reindexed, slice_ds.ReindexMapDataset)
    self.assertSequenceEqual(list(reindexed), [0, 1, 2])

  def test_via_map_dataset_slice_with_slice_unchanged(self):
    """Test that MapDataset.slice still works with a slice object."""
    ds = dataset.MapDataset.range(10)
    sliced = ds.slice(slice(2, 5))
    self.assertIsInstance(sliced, slice_ds.SliceMapDataset)
    self.assertSequenceEqual(list(sliced), [2, 3, 4])

  def test_empty_indices(self):
    ds = dataset.MapDataset.range(10)
    reindexed = slice_ds.ReindexMapDataset(ds, [])
    self.assertEmpty(reindexed)
    self.assertSequenceEqual(list(reindexed), [])


if __name__ == "__main__":
  absltest.main()
