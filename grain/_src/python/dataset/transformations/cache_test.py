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
import pickle
import shelve

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import cache as cache_dataset
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint


class CacheMapDatasetTest(absltest.TestCase):

  def test_in_memory_getitem(self):
    ds = dataset.MapDataset.range(5)
    ds = ds.map(lambda x: x**2)
    ds = cache_dataset.CacheMapDataset(ds)
    self.assertEqual(ds[0], 0)
    self.assertIsInstance(ds._cache, dict)
    assert ds._cache is not None
    self.assertIn("0", ds._cache)
    self.assertEqual(ds[1], 1)
    self.assertEqual(ds[2], 4)
    self.assertEqual(ds[3], 9)
    self.assertEqual(ds[4], 16)
    self.assertEqual(ds._cache["0"], 0)
    self.assertEqual(ds._cache["1"], 1)

  def test_file_backed_getitem(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(3)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    self.assertEqual(ds[0], 0)
    self.assertIsInstance(ds._cache, shelve.Shelf)
    assert ds._cache is not None
    self.assertEqual(ds[1], 1)
    self.assertEqual(ds[2], 2)
    ds._cache.close()  # pytype: disable=attribute-error

    ds2 = cache_dataset.CacheMapDataset(
        dataset.MapDataset.range(3), cache_path=cache_path
    )
    self.assertEqual(ds2[0], 0)
    self.assertEqual(ds2[1], 1)
    self.assertEqual(ds2[2], 2)
    ds2._cache.close()  # pytype: disable=attribute-error

  def test_file_backed_writes_file(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(3)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    self.assertEqual(ds[0], 0)
    self.assertEqual(ds[1], 1)
    self.assertEqual(ds[2], 2)
    self.assertIsInstance(ds._cache, shelve.Shelf)
    assert ds._cache is not None
    ds._cache.close()  # pytype: disable=attribute-error

    cache_filename_prefix = os.path.join(cache_path, "map_cache_worker_0")
    # shelve creates files with suffixes like .db, .dat, .bak, .dir
    # depending on implementation. We check if any file with prefix exists.
    files = [
        f for f in os.listdir(cache_path) if f.startswith("map_cache_worker_0")
    ]
    self.assertNotEmpty(files)

    # Check content
    s = shelve.open(cache_filename_prefix)
    try:
      self.assertEqual(s["0"], 0)
      self.assertEqual(s["1"], 1)
      self.assertEqual(s["2"], 2)
    finally:
      s.close()

  def test_in_memory_computation_is_cached(self):
    counter = 0

    def map_fn(x):
      nonlocal counter
      counter += 1
      return x**2

    ds = dataset.MapDataset.range(5)
    ds = ds.map(map_fn)
    ds = cache_dataset.CacheMapDataset(ds)
    self.assertEqual(ds[0], 0)
    self.assertIsInstance(ds._cache, dict)
    assert ds._cache is not None
    self.assertEqual(counter, 1)
    self.assertEqual(ds[0], 0)
    self.assertEqual(counter, 1)
    self.assertEqual(ds[1], 1)
    self.assertEqual(counter, 2)
    self.assertEqual(ds[0], 0)
    self.assertEqual(ds[1], 1)
    self.assertEqual(counter, 2)

  def test_file_backed_computation_is_cached(self):
    cache_path = self.create_tempdir().full_path
    counter = 0

    def map_fn(x):
      nonlocal counter
      counter += 1
      return x

    ds = dataset.MapDataset.range(3)
    ds = ds.map(map_fn)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    self.assertEqual(ds[0], 0)
    self.assertIsInstance(ds._cache, shelve.Shelf)
    assert ds._cache is not None
    self.assertEqual(counter, 1)
    self.assertEqual(ds[0], 0)
    self.assertEqual(counter, 1)
    self.assertEqual(ds[1], 1)
    self.assertEqual(counter, 2)
    ds._cache.close()  # pytype: disable=attribute-error

    # Create a new dataset instance to check if it reads from file cache.
    counter = 0
    ds2 = dataset.MapDataset.range(3)
    ds2 = ds2.map(map_fn)
    ds2 = cache_dataset.CacheMapDataset(ds2, cache_path=cache_path)
    self.assertEqual(ds2[0], 0)
    self.assertEqual(ds2[1], 1)
    self.assertEqual(counter, 0)
    ds2._cache.close()  # pytype: disable=attribute-error

  def test_slice_in_memory(self):
    ds = dataset.MapDataset.range(5)
    ds = cache_dataset.CacheMapDataset(ds)
    slice_ds = ds[1:4]
    self.assertLen(slice_ds, 3)
    self.assertEqual(slice_ds[0], 1)
    self.assertIsInstance(ds._cache, dict)
    assert ds._cache is not None
    self.assertEqual(slice_ds[1], 2)
    self.assertEqual(slice_ds[2], 3)
    self.assertIn("1", dict(ds._cache))
    self.assertIn("2", dict(ds._cache))
    self.assertIn("3", dict(ds._cache))

  def test_slice_file_backed(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(5)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    slice_ds = ds[1:4]
    self.assertLen(slice_ds, 3)
    self.assertEqual(slice_ds[0], 1)
    self.assertIsInstance(ds._cache, shelve.Shelf)
    assert ds._cache is not None
    self.assertEqual(slice_ds[1], 2)
    self.assertEqual(slice_ds[2], 3)
    self.assertIn("1", dict(ds._cache))
    self.assertIn("2", dict(ds._cache))
    self.assertIn("3", dict(ds._cache))

  def test_to_iter_dataset_multithreaded(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(10)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    it = ds.to_iter_dataset(
        options.ReadOptions(num_threads=2, prefetch_buffer_size=2)
    )
    self.assertCountEqual(list(it), list(range(10)))
    it2 = ds.to_iter_dataset(
        options.ReadOptions(num_threads=2, prefetch_buffer_size=1)
    )
    self.assertCountEqual(list(it2), list(range(10)))

  def test_mp_prefetch_file_backed(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(20)
    ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    ds = ds.to_iter_dataset().mp_prefetch(
        options.MultiprocessingOptions(num_workers=2)
    )
    it = ds.__iter__()
    state = it.get_state()
    self.assertEqual(list(it), list(range(20)))
    it.set_state(state)
    self.assertEqual(list(it), list(range(20)))

  def test_repeat_in_memory(self):
    counter = 0

    def map_fn(x):
      nonlocal counter
      counter += 1
      return x

    ds = dataset.MapDataset.range(3).map(map_fn)
    cache_ds = cache_dataset.CacheMapDataset(ds)
    repeated_ds = cache_ds.repeat(2, reseed_each_epoch=False)
    iter_ds = repeated_ds.to_iter_dataset(options.ReadOptions(num_threads=1))
    iterator = iter_ds.__iter__()
    saved_state = iterator.get_state()
    self.assertEqual(list(iterator), [0, 1, 2, 0, 1, 2])
    self.assertEqual(counter, 3)
    # Second iteration should use cache.
    iterator.set_state(saved_state)
    self.assertEqual(list(iterator), [0, 1, 2, 0, 1, 2])
    self.assertEqual(counter, 3)

  def test_repeat_file_backed(self):
    cache_path = self.create_tempdir().full_path
    counter = 0

    def map_fn(x):
      nonlocal counter
      counter += 1
      return x

    ds = dataset.MapDataset.range(3).map(map_fn)
    cache_ds = cache_dataset.CacheMapDataset(ds, cache_path=cache_path)
    repeated_ds = cache_ds.repeat(2, reseed_each_epoch=False)
    iter_ds = repeated_ds.to_iter_dataset(options.ReadOptions(num_threads=1))
    self.assertEqual(list(iter_ds), [0, 1, 2, 0, 1, 2])
    self.assertEqual(counter, 3)
    if cache_ds._cache:
      cache_ds._cache.close()  # pytype: disable=attribute-error

    # Create a new dataset instance to check if it reads from file cache.
    counter = 0
    ds2 = dataset.MapDataset.range(3).map(map_fn)
    cache_ds2 = cache_dataset.CacheMapDataset(ds2, cache_path=cache_path)
    repeated_ds2 = cache_ds2.repeat(2, reseed_each_epoch=False)
    iter_ds2 = repeated_ds2.to_iter_dataset(options.ReadOptions(num_threads=1))
    self.assertEqual(list(iter_ds2), [0, 1, 2, 0, 1, 2])
    self.assertEqual(counter, 0)


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
    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    self.assertEqual(list(ds), [0, 1, 2])
    cache_filename = os.path.join(cache_path, "iter_cache_worker_0")
    self.assertTrue(os.path.exists(cache_filename))
    elements = []
    with open(cache_filename, "rb") as f:
      while True:
        try:
          elements.append(pickle.load(f))
        except EOFError:
          break
    self.assertEqual(elements, [0, 1, 2])

  def test_empty_dataset_file_backed(self):
    cache_path = self.create_tempdir().full_path
    cache_path = os.path.join(cache_path, "empty_dataset")
    ds = dataset.MapDataset.range(0).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    self.assertEqual(list(ds), [])
    self.assertFalse(os.path.exists(cache_path))

  def test_file_backed_iter_loads_from_file(self):
    cache_path = self.create_tempdir().full_path
    cache_filename = os.path.join(cache_path, "iter_cache_worker_0")
    os.makedirs(cache_path, exist_ok=True)
    with open(cache_filename, "wb") as f:
      pickle.dump(10, f)
      pickle.dump(11, f)
      pickle.dump(12, f)

    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    # If cache file exists, we load from cache and shouldn't iterate through
    # parent.
    self.assertEqual(list(ds), [10, 11, 12])

  def test_file_backed_iter_two_pass(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    self.assertEqual(list(ds), [0, 1, 2])
    # The second list(ds) will create a new iterator, which should
    # read from cache file created by first list(ds) pass.
    self.assertEqual(list(ds), [0, 1, 2])

  def test_file_backed_iter_reset_before_full_iteration(self):
    cache_path = self.create_tempdir().full_path
    cache_filename = os.path.join(cache_path, "iter_cache_worker_0")
    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    iterator = iter(ds)
    self.assertEqual(next(iterator), 0)
    self.assertTrue(os.path.exists(cache_filename))
    self.assertEqual(next(iterator), 1)
    # Deleting iterator before dataset is fully read.
    # __del__ should remove cache file.
    del iterator
    gc.collect()
    self.assertFalse(os.path.exists(cache_filename))
    # New iterator will read from parent and write cache file again.
    self.assertEqual(list(ds), [0, 1, 2])
    # Now cache file should exist.
    self.assertTrue(os.path.exists(cache_filename))
    # And next iteration should use cache.
    self.assertEqual(list(ds), [0, 1, 2])

  def test_file_backed_iter_repeat_epochs(self):
    cache_path = self.create_tempdir().full_path
    counter = 0

    def increment_fn(x):
      nonlocal counter
      counter += 1
      return x

    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = ds.map(increment_fn)
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)

    # first epoch
    self.assertEqual(list(ds), [0, 1, 2])
    self.assertEqual(counter, 3)
    # second epoch - should read from cache
    self.assertEqual(list(ds), [0, 1, 2])
    self.assertEqual(counter, 3)

  def test_checkpointing_in_memory(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path="")
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_file_backed(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_in_memory_mp(self):
    ds = dataset.MapDataset.range(5).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path="")
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    assert_equal_output_after_checkpoint(ds)

  def test_checkpointing_file_backed_mp(self):
    cache_path = self.create_tempdir().full_path
    ds = dataset.MapDataset.range(5).to_iter_dataset()
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
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = cache_dataset.CacheIterDataset(ds, cache_path=cache_path)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    it = ds.__iter__()
    state = it.get_state()
    self.assertEqual(list(it), list(range(20)))
    it.set_state(state)
    self.assertEqual(list(it), list(range(20)))


if __name__ == "__main__":
  absltest.main()
