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

import gc
import os

from absl.testing import absltest
from absl.testing import parameterized
import bagz
from grain import fast_proto
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import cache as cache_dataset
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np


def _to_dict_element(x):
  return {"x": np.array(x, dtype=np.int64)}


def _compare_lists_of_dicts(list1, list2):
  for d1, d2 in zip(list1, list2):
    np.testing.assert_equal(d1, d2)


class CacheIterDatasetTest(parameterized.TestCase):

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

  def test_file_backed_iter_writes_file(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(3).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    expected = [{"x": 0}, {"x": 1}, {"x": 2}]
    actual = list(ds)
    _compare_lists_of_dicts(actual, expected)

    cache_filename = os.path.join(cache_path, "iter_cache_worker_0.bagz")
    self.assertTrue(os.path.exists(cache_filename))
    elements = [
        fast_proto.parse_tf_example_experimental(x)
        for x in bagz.BagReader(cache_filename)
    ]
    _compare_lists_of_dicts(elements, expected)

  def test_empty_dataset_file_backed(self):
    cache_path = self.create_tempdir().full_path
    cache_path = os.path.join(cache_path, "empty_dataset")
    ds = dataset.MapDataset.range(0).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    self.assertEqual(list(ds), [])
    self.assertFalse(os.path.exists(cache_path))

  def test_file_backed_iter_loads_from_file(self):
    cache_path = self.create_tempdir().full_path
    cache_filename = os.path.join(cache_path, "iter_cache_worker_0.bagz")
    os.makedirs(cache_path, exist_ok=True)
    with bagz.BagWriter(cache_filename) as writer:
      writer.write(fast_proto.make_tf_example({"x": 10}).SerializeToString())
      writer.write(fast_proto.make_tf_example({"x": 11}).SerializeToString())
      writer.write(fast_proto.make_tf_example({"x": 12}).SerializeToString())

    ds = dataset.MapDataset.range(3).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    # If cache file exists, we load from cache and shouldn't iterate through
    # parent.
    expected = [{"x": 10}, {"x": 11}, {"x": 12}]
    actual = list(ds)
    _compare_lists_of_dicts(actual, expected)

  def test_file_backed_iter_two_pass(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(3).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    expected = [{"x": 0}, {"x": 1}, {"x": 2}]
    _compare_lists_of_dicts(list(ds), expected)
    # The second list(ds) will create a new iterator, which should
    # read from cache file created by first list(ds) pass.
    _compare_lists_of_dicts(list(ds), expected)

  def test_file_backed_iter_reset_before_full_iteration(self):
    cache_path = self.create_tempdir().full_path
    cache_filename = os.path.join(cache_path, "iter_cache_worker_0.bagz")
    ds = dataset.MapDataset.range(3).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    iterator = iter(ds)
    np.testing.assert_equal(next(iterator), {"x": 0})
    self.assertTrue(os.path.exists(cache_filename))
    np.testing.assert_equal(next(iterator), {"x": 1})
    # Deleting iterator before dataset is fully read.
    # __del__ should remove cache file.
    del iterator
    gc.collect()
    self.assertFalse(os.path.exists(cache_filename))
    # New iterator will read from parent and write cache file again.
    expected = [{"x": 0}, {"x": 1}, {"x": 2}]
    _compare_lists_of_dicts(list(ds), expected)
    # Now cache file should exist.
    self.assertTrue(os.path.exists(cache_filename))
    # And next iteration should use cache.
    _compare_lists_of_dicts(list(ds), expected)

  def test_file_backed_iter_repeat_epochs(self):
    cache_path = self.create_tempdir().full_path
    counter = 0

    def increment_fn(x):
      nonlocal counter
      counter += 1
      return _to_dict_element(x)

    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = ds.map(increment_fn)
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    expected = [{"x": 0}, {"x": 1}, {"x": 2}]

    # first epoch
    _compare_lists_of_dicts(list(ds), expected)
    self.assertEqual(counter, 3)
    # second epoch - should read from cache
    _compare_lists_of_dicts(list(ds), expected)
    self.assertEqual(counter, 3)

  def test_checkpointing_in_memory(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path="")
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_file_backed(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(5).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_in_memory_mp(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path="")
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_file_backed_mp(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(5).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    assert_equal_output_after_checkpoint(ds)

  def test_mp_prefetch_in_memory(self):
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path="")
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    it = ds.__iter__()
    state = it.get_state()
    self.assertEqual(list(it), list(range(20)))
    it.set_state(state)
    self.assertEqual(list(it), list(range(20)))

  def test_mp_prefetch_file_backed(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(20).map(_to_dict_element).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    it = ds.__iter__()
    state = it.get_state()
    expected = [{"x": i} for i in range(20)]
    _compare_lists_of_dicts(list(it), expected)
    it.set_state(state)
    _compare_lists_of_dicts(list(it), expected)


if __name__ == "__main__":
  absltest.main()
