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


if __name__ == "__main__":
  absltest.main()
