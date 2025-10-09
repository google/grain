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
"""Tests for repeat transformation."""
import sys

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import repeat
from grain._src.python.testing import experimental as testing
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


class RepeatMapDatasetTest(parameterized.TestCase):

  def test_finite_num_epochs_changes_length(self):
    ds = dataset.MapDataset.range(6)
    self.assertLen(ds, 6)
    ds = repeat.RepeatMapDataset(ds, num_epochs=3)
    self.assertLen(ds, 18)

  def test_finite_num_epochs_produces_expected_elements_when_iterated(self):
    ds = dataset.MapDataset.range(4)
    ds = repeat.RepeatMapDataset(ds, num_epochs=3)
    self.assertSequenceEqual(list(ds), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

  def test_infinite_epochs_produces_expected_elements_when_iterated(self):
    ds = dataset.MapDataset.range(4)
    ds = repeat.RepeatMapDataset(ds, num_epochs=None)
    self.assertSequenceEqual(
        ds[0 : 4 * 3], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    )

  def test_infinite_epochs_sets_length_to_maxsize(self):
    ds = dataset.MapDataset.range(6)
    ds = repeat.RepeatMapDataset(ds, num_epochs=None)
    self.assertLen(ds, sys.maxsize)

  def test_repeat_after_setting_infinite_epochs_raises_value_error(self):
    ds = dataset.MapDataset.range(6)
    ds = repeat.RepeatMapDataset(ds, num_epochs=None)
    with self.assertRaises(ValueError):
      repeat.RepeatMapDataset(ds, num_epochs=2)

  def test_setting_zero_epochs_raises_value_error(self):
    ds = dataset.MapDataset.range(6)
    with self.assertRaises(ValueError):
      repeat.RepeatMapDataset(ds, num_epochs=0)

  def test_setting_negative_epochs_raises_value_error(self):
    ds = dataset.MapDataset.range(6)
    with self.assertRaises(ValueError):
      repeat.RepeatMapDataset(ds, num_epochs=-1)

  def test_infinite_epochs_of_empty_dataset_keeps_length_zero(self):
    ds = EmptyMapDataset()
    ds = repeat.RepeatMapDataset(ds, num_epochs=None)
    self.assertEmpty(ds)

  def test_finite_num_empochs_get_items_produces_expected_elements(self):
    ds = dataset.MapDataset.range(4)
    ds = repeat.RepeatMapDataset(ds, num_epochs=3)
    self.assertSequenceEqual(
        ds._getitems(list(range(4 * 3))), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    )
    self.assertSequenceEqual(
        ds._getitems(list([0, 1, 4, 5, 8, 9])), [0, 1, 0, 1, 0, 1]
    )

  def test_infinite_epochs_get_items_produces_expected_elements(self):
    ds = dataset.MapDataset.range(4)
    ds = repeat.RepeatMapDataset(ds, num_epochs=None)
    self.assertSequenceEqual(
        ds._getitems(list(range(4 * 3))), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    )
    self.assertSequenceEqual(
        ds._getitems(list([0, 1, 4, 5, 8, 9])), [0, 1, 0, 1, 0, 1]
    )

  @parameterized.product(
      num_epochs=[2, None],
      reseed_each_epoch=[True, False],
      explicit_seed=[True, False],
      use_batched_read=[True, False],
  )
  def test_random_transforms_across_epochs(
      self,
      num_epochs: int | None,
      reseed_each_epoch: bool,
      explicit_seed: bool,
      use_batched_read: bool,
  ):
    num_examples = 1000
    ds = dataset.MapDataset.range(num_examples)
    if explicit_seed:
      ds = ds.shuffle(seed=42).random_map(
          lambda x, rng: x + rng.integers(10), seed=43
      )
    else:
      ds = ds.seed(42).shuffle().random_map(lambda x, rng: x + rng.integers(10))
    ds = repeat.RepeatMapDataset(
        ds, num_epochs=num_epochs, reseed_each_epoch=reseed_each_epoch
    )

    if use_batched_read:
      first_epoch = ds._getitems(list(range(num_examples)))
      second_epoch = ds._getitems(list(range(num_examples, num_examples * 2)))
    else:
      first_epoch = list(ds[:num_examples])
      second_epoch = list(ds[num_examples : num_examples * 2])

    if reseed_each_epoch:
      self.assertNotEqual(first_epoch, second_epoch)
      self.assertNotEqual(set(first_epoch), set(second_epoch))
    else:
      self.assertEqual(first_epoch, second_epoch)


class RepeatIterDatasetTest(parameterized.TestCase):

  def test_finite_num_epochs_produces_expected_elements(self):
    ds = dataset.MapDataset.range(4).to_iter_dataset()
    ds = repeat.RepeatIterDataset(ds, num_epochs=3)
    self.assertSequenceEqual(list(ds), [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

  def test_infinite_epochs_produces_expected_elements(self):
    ds = dataset.MapDataset.range(4).to_iter_dataset()
    ds = repeat.RepeatIterDataset(ds, num_epochs=None)
    results = []
    for i, val in enumerate(ds):
      if i == 12:
        break
      results.append(val)
    self.assertSequenceEqual(results, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

  def test_setting_zero_epochs_raises_value_error(self):
    ds = dataset.MapDataset.range(6).to_iter_dataset()
    with self.assertRaises(ValueError):
      repeat.RepeatIterDataset(ds, num_epochs=0)

  def test_setting_negative_epochs_raises_value_error(self):
    ds = dataset.MapDataset.range(6).to_iter_dataset()
    with self.assertRaises(ValueError):
      repeat.RepeatIterDataset(ds, num_epochs=-1)

  def test_repeat_empty_dataset_is_empty(self):
    ds = EmptyMapDataset().to_iter_dataset()
    ds = repeat.RepeatIterDataset(ds, num_epochs=2)
    self.assertEmpty(list(ds))

  @parameterized.parameters(1, 2, 5)
  def test_checkpointing(self, num_epochs):
    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = repeat.RepeatIterDataset(ds, num_epochs=num_epochs)
    testing.assert_equal_output_after_checkpoint(ds)


if __name__ == "__main__":
  absltest.main()
