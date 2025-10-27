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
"""Tests for cache transformations."""

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import cache as cache_dataset
from grain._src.python.dataset.transformations import limit as limit_dataset
from grain._src.python.dataset.transformations import repeat as repeat_dataset
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint


class InMemoryCacheIterDatasetTest(parameterized.TestCase):

  def test_in_memory_iter(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = ds.map(lambda x: x**2)
    ds = cache_dataset.CacheIterDataset(ds)
    self.assertEqual(list(ds), [0, 1, 4, 9, 16])

  def test_empty_dataset_in_memory(self):
    ds = dataset.MapDataset.range(0).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds)
    self.assertEqual(list(ds), [])
    self.assertEqual(list(ds), [])

  def test_checkpointing_in_memory(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds)
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_in_memory_mp(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    assert_equal_output_after_checkpoint(ds)

  def test_mp_prefetch_in_memory(self):
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    it = ds.__iter__()
    state = it.get_state()
    self.assertEqual(list(it), list(range(20)))
    it.set_state(state)
    self.assertEqual(list(it), list(range(20)))

  def test_in_memory_with_limit_and_repeat(self):
    counter = 0

    def increment_fn(x):
      nonlocal counter
      counter += 1
      return x

    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = ds.map(increment_fn)
    ds = limit_dataset.LimitIterDataset(ds, 10)
    ds = cache_dataset.CacheIterDataset(ds)
    ds = repeat_dataset.RepeatIterDataset(ds, num_epochs=2)

    expected = list(range(10)) + list(range(10))
    self.assertEqual(list(ds), expected)
    self.assertEqual(counter, 10)

  def test_checkpointing_in_memory_with_limit_and_repeat(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    ds = limit_dataset.LimitIterDataset(ds, 5)
    ds = cache_dataset.CacheIterDataset(ds)
    ds = repeat_dataset.RepeatIterDataset(ds, num_epochs=2)
    it = ds.__iter__()
    for _ in range(7):
      next(it)
    state = it.get_state()
    remaining = list(it)
    it = ds.__iter__()
    it.set_state(state)
    self.assertEqual(list(it), remaining)
    self.assertEqual(remaining, [2, 3, 4])


if __name__ == "__main__":
  absltest.main()
