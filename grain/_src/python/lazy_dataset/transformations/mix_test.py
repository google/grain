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
"""Tests for mixing transformation."""

import sys
from typing import Callable, Tuple

from absl.testing import absltest
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import mix
from grain._src.python.lazy_dataset.transformations import repeat  # pylint: disable=unused-import
import numpy as np


class ExplicitSelectionMap(mix.DatasetSelectionMap):

  def __init__(
      self, length: int, selection_map: Callable[[int], Tuple[int, int]]
  ):
    self._length = length
    self._selection_map = selection_map

  def __len__(self) -> int:
    return self._length

  def __getitem__(self, index: int) -> Tuple[int, int]:
    return self._selection_map(index)


class MixedLazyMapTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.even = range(0, 10, 2)
    self.odd = range(1, 10, 2)

  def test_interleaved_map(self):
    expected_indices = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (0, 4),
        (1, 4),
    ]
    expected_dataset = list(range(10))

    def _inteleaved_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      ds = index % 2
      ds_index = index // 2
      return (ds, ds_index)

    interleaved_map = ExplicitSelectionMap(10, _inteleaved_dataset)

    components = [self.even, self.odd]
    indices = [interleaved_map[i] for i in range(10)]
    unrolled_dataset = [components[ds][ds_index] for ds, ds_index in indices]

    self.assertLen(interleaved_map, 10)
    self.assertEqual(expected_indices, indices)
    self.assertEqual(expected_dataset, unrolled_dataset)

  def test_sequential_map(self):
    expected_indices = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]
    expected_dataset = list(self.even) + list(self.odd)

    def _sequential_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      if index < 5:
        ds = 0
      else:
        ds = 1
      ds_index = index % 5
      return (ds, ds_index)

    sequential_map = ExplicitSelectionMap(10, _sequential_dataset)

    components = [self.even, self.odd]
    indices = [sequential_map[i] for i in range(10)]
    unrolled_dataset = [components[ds][ds_index] for ds, ds_index in indices]

    self.assertLen(sequential_map, 10)
    self.assertEqual(expected_indices, indices)
    self.assertEqual(expected_dataset, unrolled_dataset)

  def test_subset_and_shuffle_map(self):
    first_epoch = [0, 1, 2, 3, 4]
    second_epoch = [1, 0, 3, 2, 4]
    expected_indices = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 0),
        (0, 0),
        (1, 1),
        (0, 1),
        (0, 2),
    ]
    expected_dataset = first_epoch + second_epoch

    def _subset_and_shuffle_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      if index < 5:
        ds = index % 2
        ds_index = index // 2
      else:
        mapped_index = second_epoch[index - 5]
        ds = mapped_index % 2
        ds_index = mapped_index // 2
      return (ds, ds_index)

    subset_and_shuffle_map = ExplicitSelectionMap(
        10, _subset_and_shuffle_dataset
    )

    components = [self.even, self.odd]
    indices = [subset_and_shuffle_map[i] for i in range(10)]
    unrolled_dataset = [components[ds][ds_index] for ds, ds_index in indices]

    self.assertLen(subset_and_shuffle_map, 10)
    self.assertEqual(expected_indices, indices)
    self.assertEqual(expected_dataset, unrolled_dataset)


class MixedLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.even_ds = lazy_dataset.RangeLazyMapDataset(0, 10, 2)
    self.odd_ds = lazy_dataset.RangeLazyMapDataset(1, 10, 2)

  def test_len(self):
    # Mix dataset has length to see any element at most once.
    ds1 = lazy_dataset.RangeLazyMapDataset(10)
    ds2 = lazy_dataset.RangeLazyMapDataset(20)
    ds3 = lazy_dataset.RangeLazyMapDataset(5)
    # Equal proportions.
    ds = mix.MixedLazyMapDataset([ds1, ds2, ds3])
    self.assertLen(ds, 15)
    # Heigher weight for second dataset.
    ds = mix.MixedLazyMapDataset([ds1, ds2, ds3], proportions=[1, 2, 1])
    self.assertLen(ds, 5 + 10 + 5)

  def test_mixing_equal_probability_with_integer_proportions(self):
    mixed_lzds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[2, 2]
    )
    actual_values = [mixed_lzds[i] for i in range(10)]
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_equal_probability_with_float_proportions(self):
    mixed_lzds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[0.5, 0.5]
    )
    actual_values = [mixed_lzds[i] for i in range(10)]
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_equal_probability_with_no_proportions_given(self):
    mixed_lzds = mix.MixedLazyMapDataset(parents=[self.even_ds, self.odd_ds])
    # If no proportions specified, parents are mixed in equal proportions.
    actual_values = [mixed_lzds[i] for i in range(10)]
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_with_float_proportions(self):
    mixed_lzds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[0.75, 0.25]
    )
    self.assertLen(mixed_lzds, 6)

    actual_vals = list(mixed_lzds)
    expected_frist_epoch = [0, 2, 4, 1, 6, 8]
    self.assertEqual(actual_vals, expected_frist_epoch)

    actual_vals = list(mixed_lzds.repeat(2))
    expected_two_epochs = [
        0,
        2,
        4,
        1,
        6,
        8,
        0,
        3,
        2,
        4,
        6,
        5,
    ]
    self.assertEqual(actual_vals, expected_two_epochs)

  def test_mixing_with_integer_proportions(self):
    mixed_lzds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[1, 2]
    )
    self.assertLen(list(mixed_lzds), 7)

    actual_values = list(mixed_lzds)
    expected_first_epoch = [0, 1, 3, 2, 5, 7, 4]
    self.assertEqual(expected_first_epoch, actual_values)

    actual_values = list(mixed_lzds.repeat(2))
    expected_two_epochs = [0, 1, 3, 2, 5, 7, 4, 9, 1, 6, 3, 5, 8, 7]
    self.assertEqual(expected_two_epochs, actual_values)

  def test_mixing_zero_one_probability_fails_with_error(self):
    with self.assertRaises(ValueError):
      _ = mix.MixedLazyMapDataset(
          parents=[self.even_ds, self.odd_ds], proportions=[0, 1]
      )

  def test_mix_infinite_datasets(self):
    zeros = lazy_dataset.RangeLazyMapDataset(0, 1).repeat()
    ones = lazy_dataset.RangeLazyMapDataset(1, 2).repeat()
    self.assertLen(zeros, sys.maxsize)
    self.assertLen(ones, sys.maxsize)
    ld = mix.MixedLazyMapDataset([zeros, ones], proportions=[4, 1])
    self.assertLen(ld, sys.maxsize)
    # Mix again.
    ld = mix.MixedLazyMapDataset([ld, ones], proportions=[1, 1])
    num_samples = 1000
    value_counts = np.bincount([ld[i] for i in range(num_samples)]).tolist()
    self.assertEqual(value_counts, [400, 600])

  def test_interleaved_map(self):
    expected_dataset = list(range(10))

    def _inteleaved_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      ds = index % 2
      ds_index = index // 2
      return (ds, ds_index)

    interleaved_map = ExplicitSelectionMap(10, _inteleaved_dataset)

    ds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], selection_map=interleaved_map
    )

    self.assertEqual(list(ds), expected_dataset)

  def test_sequential_map(self):
    expected_dataset = list(range(0, 10, 2)) + list(range(1, 10, 2))

    def _sequential_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      if index < 5:
        ds = 0
      else:
        ds = 1
      ds_index = index % 5
      return (ds, ds_index)

    sequential_map = ExplicitSelectionMap(10, _sequential_dataset)

    ds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], selection_map=sequential_map
    )

    self.assertEqual(list(ds), expected_dataset)

  def test_subset_and_shuffle_map(self):
    first_epoch = [0, 1, 2, 3, 4]
    second_epoch = [1, 0, 3, 2, 4]

    expected_dataset = first_epoch + second_epoch

    def _subset_and_shuffle_dataset(index):
      if index > 9:
        raise IndexError("index our of range")
      if index < 5:
        ds = index % 2
        ds_index = index // 2
      else:
        mapped_index = second_epoch[index - 5]
        ds = mapped_index % 2
        ds_index = mapped_index // 2
      return (ds, ds_index)

    subset_and_shuffle_map = ExplicitSelectionMap(
        10, _subset_and_shuffle_dataset
    )

    ds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds],
        selection_map=subset_and_shuffle_map,
    )

    self.assertEqual(list(ds), expected_dataset)


class MixedLazyIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.even_map_ds = lazy_dataset.RangeLazyMapDataset(0, 10, 2)
    self.odd_map_ds = lazy_dataset.RangeLazyMapDataset(1, 10, 2)
    self.even_ds = self.even_map_ds.to_iter_dataset()
    self.odd_ds = self.odd_map_ds.to_iter_dataset()

  def test_len(self):
    # Mixed dataset sees any element at most once.
    ds1 = lazy_dataset.RangeLazyMapDataset(10).to_iter_dataset()
    ds2 = lazy_dataset.RangeLazyMapDataset(20).to_iter_dataset()
    ds3 = lazy_dataset.RangeLazyMapDataset(5).to_iter_dataset()
    # Equal proportions.
    ds = mix.MixedLazyIterDataset([ds1, ds2, ds3])
    # While ds3 is empty after sampling 15 elements, a StopIteration is raised
    # when an example is sampled from ds3 the next time; an example is sampled
    # from ds1 and ds2 before then; hence the size of the mixed dataset is 17
    # instead of 15.
    self.assertLen(list(ds), 17)
    # Heigher weight for second dataset.
    ds = mix.MixedLazyIterDataset([ds1, ds2, ds3], proportions=[1, 2, 1])
    # ds1 is sampled from once and ds2 is sampled from twice before
    # StopIteration is raised by ds3
    self.assertLen(list(ds), 6 + 12 + 5)

  def test_mixing_equal_probability_with_integer_proportions(self):
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[2, 2]
    )
    actual_values = list(mixed_lzds)
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_equal_probability_with_float_proportions(self):
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[0.5, 0.5]
    )
    actual_values = list(mixed_lzds)
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_equal_probability_with_no_proportions_given(self):
    mixed_lzds = mix.MixedLazyIterDataset(parents=[self.even_ds, self.odd_ds])
    # If no proportions specified, parents are mixed in equal proportions.
    actual_values = list(mixed_lzds)
    expected_values = [val for val in range(10)]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_with_float_proportions(self):
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[0.75, 0.25]
    )
    actual_vals = list(mixed_lzds)
    expected_frist_epoch = [0, 2, 4, 1, 6, 8]
    self.assertEqual(actual_vals, expected_frist_epoch)

    # Mix with repeats.
    ds1 = self.even_map_ds.repeat(2).to_iter_dataset()
    ds2 = self.odd_map_ds.repeat(2).to_iter_dataset()
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[ds1, ds2], proportions=[0.75, 0.25]
    )
    actual_vals = list(mixed_lzds)
    expected_two_epochs = [0, 2, 4, 1, 6, 8, 0, 3, 2, 4, 6, 5, 8]
    self.assertEqual(actual_vals, expected_two_epochs)

  def test_mixing_with_integer_proportions(self):
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[1, 2]
    )
    self.assertLen(list(mixed_lzds), 8)

    actual_values = list(mixed_lzds)
    expected_first_epoch = [0, 1, 3, 2, 5, 7, 4, 9]
    self.assertEqual(expected_first_epoch, actual_values)

    # Mix with repeats.
    ds1 = self.even_map_ds.repeat(2).to_iter_dataset()
    ds2 = self.odd_map_ds.repeat(2).to_iter_dataset()
    mixed_lzds = mix.MixedLazyIterDataset(
        parents=[ds1, ds2], proportions=[1, 2]
    )
    actual_values = list(mixed_lzds)
    expected_two_epochs = [0, 1, 3, 2, 5, 7, 4, 9, 1, 6, 3, 5, 8, 7, 9, 0]
    self.assertEqual(expected_two_epochs, actual_values)

  def test_mixing_zero_one_probability_fails_with_error(self):
    with self.assertRaises(ValueError):
      _ = mix.MixedLazyIterDataset(
          parents=[self.even_ds, self.odd_ds], proportions=[0, 1]
      )

  def test_mix_infinite_datasets(self):
    zeros = lazy_dataset.RangeLazyMapDataset(0, 1).repeat()
    ones = lazy_dataset.RangeLazyMapDataset(1, 2).repeat()
    self.assertLen(zeros, sys.maxsize)
    self.assertLen(ones, sys.maxsize)
    ld = mix.MixedLazyIterDataset(
        [zeros.to_iter_dataset(), ones.to_iter_dataset()], proportions=[4, 1]
    )
    # Mix again.
    ld = mix.MixedLazyIterDataset(
        [ld, ones.to_iter_dataset()], proportions=[1, 1]
    )
    ld_iter = iter(ld)
    num_samples = 1000
    value_counts = np.bincount(
        [next(ld_iter) for _ in range(num_samples)]
    ).tolist()
    self.assertEqual(value_counts, [400, 600])

  def test_stop_sampling_after_end_of_any_dataset(self):
    smaller_ds = lazy_dataset.RangeLazyMapDataset(5).to_iter_dataset()
    larger_ds = lazy_dataset.RangeLazyMapDataset(10).to_iter_dataset()
    mixed_ds = mix.MixedLazyIterDataset([smaller_ds, larger_ds], [1.0, 1.0])
    ds_iter = iter(mixed_ds)
    for _ in range(10):  # Exhaust the iterator.
      _ = next(ds_iter)
    for _ in range(10):  # Verify that no more examples are sampled.
      with self.assertRaises(StopIteration):
        _ = next(ds_iter)

  def test_checkpointing(self):
    ds = mix.MixedLazyIterDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[1, 1]
    )
    ds_iter = iter(ds)

    max_steps = 10
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 1, 5, 8]:
      ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        np.testing.assert_array_equal(
            next(ds_iter), values_without_interruption[i]
        )


if __name__ == "__main__":
  absltest.main()
