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
"""Tests for LazyDataset DataLoader."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python.lazy_dataset import lazy_dataset_data_loader
from grain.python_experimental import lazy_dataset
import numpy as np


# Functions needs to be defined at the top level in order to be picklable.


@dataclasses.dataclass(frozen=True)
class NonPicklableTransform(transforms.MapTransform):

  def __getstate__(self):
    raise ValueError("This transformation cannot be pickled.")

  def map(self, element):
    return element


@dataclasses.dataclass(frozen=True)
class SimpleMapTransform(transforms.MapTransform):

  def map(self, element: int) -> int:
    return element


class SimpleRandomMapTransform(transforms.RandomMapTransform):

  def random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + rng.integers(100_000)


class LazyDatasetDataLoaderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_src = lazy_dataset.RangeLazyMapDataset(start=0, stop=10, step=1)
    self.lazy_ds = lazy_dataset.MapLazyMapDataset(
        parent=self.ds_src, transform=SimpleMapTransform()
    )

  def test_data_loader_produces_correct_single_elements(self):
    data_loader = lazy_dataset_data_loader.LazyDatasetDataLoader(
        lazy_ds=self.lazy_ds
    )
    output_elements = list(iter(data_loader))
    expected_elements = list(range(10))
    np.testing.assert_array_equal(expected_elements, output_elements)

  def test_data_loader_produces_correct_batched_elements(self):
    batched_lazy_iter_ds = lazy_dataset.BatchLazyIterDataset(
        parent=self.lazy_ds,
        batch_size=2,
    )
    data_loader = lazy_dataset_data_loader.LazyDatasetDataLoader(
        lazy_ds=batched_lazy_iter_ds,
    )
    output_elements = list(iter(data_loader))
    expected_elements = [np.array([i, i + 1]) for i in range(0, 10, 2)]
    np.testing.assert_array_equal(expected_elements, output_elements)

  def test_data_loader_enter_and_exit_context(self):
    with lazy_dataset_data_loader.LazyDatasetDataLoader(
        lazy_ds=self.lazy_ds,
    ) as data_loader:
      output_elements = list(iter(data_loader))
    expected_elements = list(range(10))
    np.testing.assert_array_equal(expected_elements, output_elements)

  def test_data_loader_iterates_one_epoch(self):
    data_loader = lazy_dataset_data_loader.LazyDatasetDataLoader(
        lazy_ds=self.lazy_ds,
    )
    ds_iter = iter(data_loader)
    _ = [next(ds_iter) for _ in range(10)]

    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_data_loader_non_picklable_transform_raises_error(self):
    self.ds_src = lazy_dataset.RangeLazyMapDataset(start=0, stop=10, step=1)
    lazy_ds = lazy_dataset.MapLazyMapDataset(
        parent=self.ds_src, transform=NonPicklableTransform()
    )
    with self.assertRaises(ValueError):
      _ = lazy_dataset_data_loader.LazyDatasetDataLoader(lazy_ds=lazy_ds)

  def test_data_loader_invalid_number_of_workers_raises_error(self):
    with self.assertRaises(ValueError):
      _ = lazy_dataset_data_loader.LazyDatasetDataLoader(
          lazy_ds=self.lazy_ds,
          num_processes=-1,
      )

  def test_data_loader_with_random_map_transform(self):
    lazy_ds = lazy_dataset.MapLazyMapDataset(
        parent=self.ds_src, transform=SimpleRandomMapTransform(), seed=0
    )
    data_loader = lazy_dataset_data_loader.LazyDatasetDataLoader(
        lazy_ds=lazy_ds,
    )
    ds_iter = iter(data_loader)
    _ = [next(ds_iter) for _ in range(10)]


if __name__ == "__main__":
  absltest.main()
