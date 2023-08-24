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

from absl.testing import absltest
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import mix


class MixedLazyMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.even_ds = lazy_dataset.RangeLazyMapDataset(0, 10, 2)
    self.odd_ds = lazy_dataset.RangeLazyMapDataset(1, 10, 2)

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
    actual_vals = [mixed_lzds[i] for i in range(len(mixed_lzds))]
    expected_vals = [0, 2, 4, 1, 6, 8, 0, 3, 2, 4, 6, 5, 8, 0, 2, 7, 4, 6, 8, 9]
    self.assertEqual(expected_vals, actual_vals)

  def test_mixing_with_integer_proportions(self):
    mixed_lzds = mix.MixedLazyMapDataset(
        parents=[self.even_ds, self.odd_ds], proportions=[1, 2]
    )
    actual_values = [mixed_lzds[i] for i in range(len(mixed_lzds))]
    expected_values = [0, 1, 3, 2, 5, 7, 4, 9, 1, 6, 3, 5, 8, 7, 9]
    self.assertEqual(expected_values, actual_values)

  def test_mixing_zero_one_probability_fails_with_error(self):
    with self.assertRaises(ValueError):
      _ = mix.MixedLazyMapDataset(
          parents=[self.even_ds, self.odd_ds], proportions=[0, 1]
      )


if __name__ == "__main__":
  absltest.main()
