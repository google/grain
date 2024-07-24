# Copyright 2024 Google LLC
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
"""Tests for prefetch.py."""

import dataclasses
import time
from typing import TypeVar, cast
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import filter as filter_lazy_dataset
from grain._src.python.dataset.transformations import map as map_lazy_dataset
from grain._src.python.dataset.transformations import prefetch


_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True)
class FilterKeepingOddElementsOnly(transforms.FilterTransform):

  def filter(self, element: int) -> bool:
    return bool(element % 2)


class PrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(20)
    self.filtered_range_ds = filter_lazy_dataset.FilterMapDataset(
        self.range_ds, FilterKeepingOddElementsOnly()
    )
    self.prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions()
    )

  def test_dataset_and_iterator_types(self):
    self.assertIsInstance(
        self.prefetch_lazy_iter_ds, prefetch.PrefetchIterDataset
    )
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)

  @parameterized.parameters(0, 1, 10)
  def test_prefetch_data_dense(self, prefetch_buffer_size: int):
    read_options = options.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size
    )
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=read_options
    )
    self.assertEqual(prefetch_lazy_iter_ds._read_options, read_options)  # pylint: disable=protected-access
    ds_iter = iter(prefetch_lazy_iter_ds)
    actual = [next(ds_iter) for _ in range(20)]
    expected = list(range(20))
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(0, 1, 10)
  def test_prefetch_data_sparse(self, prefetch_buffer_size: int):
    read_options = options.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size
    )
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.filtered_range_ds,
        read_options=read_options,
        allow_nones=True,
    )
    self.assertEqual(prefetch_lazy_iter_ds._read_options, read_options)  # pylint: disable=protected-access
    ds_iter = iter(prefetch_lazy_iter_ds)
    actual = [next(ds_iter) for _ in range(20)]
    expected = [i if i % 2 == 1 else None for i in range(20)]
    self.assertSequenceEqual(actual, expected)

  def test_prefetch_iterates_one_epoch(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    _ = [next(ds_iter) for _ in range(20)]
    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_prefetch_does_not_buffer_unnecessary_elements(self):
    prefetch_buffer_size = 15
    prefetch_lazy_iter_ds_large_buffer = prefetch.PrefetchIterDataset(
        self.range_ds,
        read_options=options.ReadOptions(
            prefetch_buffer_size=prefetch_buffer_size
        ),
    )
    ds_iter = iter(prefetch_lazy_iter_ds_large_buffer)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)
    self.assertIsNone(ds_iter._buffer)
    _ = next(ds_iter)
    self.assertLen(ds_iter._buffer, prefetch_buffer_size)
    _ = [next(ds_iter) for _ in range(14)]
    self.assertLen(
        ds_iter._buffer, len(self.range_ds) - prefetch_buffer_size
    )  # iterated through 15 elements so far
    _ = [next(ds_iter) for _ in range(5)]
    self.assertEmpty(ds_iter._buffer)  # iterated through all elements

  def test_checkpoint(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)

    max_steps = 20
    values_without_interruption = []
    checkpoints = []
    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 1, 5, 12, 18]:
      ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])

  @parameterized.parameters(-1, 30)
  def test_invalid_checkpoint(self, next_index: int):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    with self.assertRaisesRegex(
        IndexError,
        f'Checkpoint `next_index` {next_index} is out of range for dataset of'
        ' length 20.',
    ):
      ds_iter.set_state({'next_index': next_index})  # pytype: disable=attribute-error


class MultiprocessPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    ds = dataset.MapDataset.range(20)
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    self.iter_ds = filter_lazy_dataset.FilterIterDataset(
        ds, FilterKeepingOddElementsOnly()
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='1_worker',
          num_workers=1,
          per_worker_buffer_size=1,
      ),
      dict(
          testcase_name='1_worker_large_buffer',
          num_workers=1,
          per_worker_buffer_size=20,
      ),
      dict(
          testcase_name='10_workers',
          num_workers=10,
          per_worker_buffer_size=1,
      ),
      dict(
          testcase_name='10_workers_large_buffer',
          num_workers=10,
          per_worker_buffer_size=20,
      ),
  )
  def test_prefetch_data(self, num_workers: int, per_worker_buffer_size: int):
    prefetch_lazy_iter_ds = prefetch.MultiprocessPrefetchIterDataset(
        self.iter_ds,
        options.MultiprocessingOptions(num_workers, per_worker_buffer_size),
    )
    actual = list(prefetch_lazy_iter_ds)
    expected = list(range(1, 20, 2))
    self.assertSequenceEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='1_worker',
          num_workers=7,
          record_state_interval=prefetch._RECORD_STATE_INTERVAL_S,
      ),
      dict(
          testcase_name='10_workers',
          num_workers=10,
          record_state_interval=prefetch._RECORD_STATE_INTERVAL_S,
      ),
      dict(
          testcase_name='10_workers_with_continuous_state_recording',
          num_workers=10,
          record_state_interval=0,
      ),
  )
  def test_checkpoint(self, num_workers: int, record_state_interval: int):
    with mock.patch.object(
        prefetch, '_RECORD_STATE_INTERVAL_S', record_state_interval
    ):
      ds = prefetch.MultiprocessPrefetchIterDataset(
          self.iter_ds,
          options.MultiprocessingOptions(num_workers),
      )
      ds_iter = iter(ds)

      max_steps = 10
      values_without_interruption = []
      checkpoints = []
      for _ in range(max_steps):
        checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
        values_without_interruption.append(next(ds_iter))

      for starting_step in [0, 3, 8]:
        ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
        for i in range(starting_step, max_steps):
          value = next(ds_iter)
          self.assertEqual(value, values_without_interruption[i])

  def test_fails_with_0_workers(self):
    with self.assertRaisesRegex(
        ValueError, '`num_workers` must be greater than 0'
    ):
      prefetch.MultiprocessPrefetchIterDataset(
          self.iter_ds,
          options.MultiprocessingOptions(),
      )

  def test_fails_with_multiple_prefetches(self):
    ds = prefetch.MultiprocessPrefetchIterDataset(
        self.iter_ds,
        options.MultiprocessingOptions(num_workers=10),
    )
    with self.assertRaisesRegex(
        ValueError,
        'Having multiple `MultiprocessPrefetchIterDataset`s is not allowed.',
    ):
      _ = prefetch.MultiprocessPrefetchIterDataset(
          ds,
          options.MultiprocessingOptions(num_workers=1),
      )

  @parameterized.product(
      start_prefetch_calls=[0, 1, 10],
      num_workers=[6],
      per_worker_buffer_size=[1, 20],
  )
  def test_start_prefetch(
      self,
      start_prefetch_calls: int,
      num_workers: int,
      per_worker_buffer_size: int,
  ):
    class _SleepTransform(transforms.MapTransform):

      def map(self, features):
        time.sleep(1)
        return features

    ds = dataset.MapDataset.range(10)
    ds = map_lazy_dataset.MapMapDataset(parent=ds, transform=_SleepTransform())
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    ds = prefetch.MultiprocessPrefetchIterDataset(
        ds,
        options.MultiprocessingOptions(num_workers, per_worker_buffer_size),
    )

    it = iter(ds)
    assert isinstance(it, prefetch.MultiprocessPrefetchDatasetIterator)
    for _ in range(start_prefetch_calls):
      it.start_prefetch()

    # Waits for prefetching.
    start_time = time.time()
    while time.time() - start_time < 30:
      time.sleep(2)

    # Measures time to read from the dataset.
    start_time = time.time()
    self.assertSequenceEqual(list(it), list(range(10)))

    time_to_fetch = time.time() - start_time
    logging.info('Reading dataset took %.2f seconds.', time_to_fetch)
    if start_prefetch_calls:
      self.assertLess(time_to_fetch, 5)
    else:
      self.assertGreater(time_to_fetch, 1)

  def test_prefetch_but_no_read(self):
    class _SleepTransform(transforms.MapTransform):

      def map(self, features):
        time.sleep(1)
        return features

    ds = dataset.MapDataset.range(10)
    ds = map_lazy_dataset.MapMapDataset(parent=ds, transform=_SleepTransform())
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    ds = prefetch.MultiprocessPrefetchIterDataset(
        ds,
        options.MultiprocessingOptions(
            num_workers=3, per_worker_buffer_size=20
        ),
    )

    # Makes sure the iterator cleans up gracefully if it is prefetched but no
    # elements are read.
    it = iter(ds)
    assert isinstance(it, prefetch.MultiprocessPrefetchDatasetIterator)
    it.start_prefetch()
    # Waits for the processes to actually read some elements and put them into
    # buffers.
    time.sleep(30)


class ThreadPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds = filter_lazy_dataset.FilterIterDataset(
        dataset.MapDataset.range(20).to_iter_dataset(),
        FilterKeepingOddElementsOnly(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='thread',
          prefetch_buffer_size=1,
          warm_start=True,
      ),
      dict(
          testcase_name='thread_large_buffer',
          prefetch_buffer_size=20,
          warm_start=False,
      ),
      dict(
          testcase_name='thread_huge_buffer',
          prefetch_buffer_size=200,
          warm_start=True,
      ),
  )
  def test_prefetch_data(self, prefetch_buffer_size: int, warm_start: bool):
    prefetch_lazy_iter_ds = prefetch.ThreadPrefetchIterDataset(
        self.ds, prefetch_buffer_size=prefetch_buffer_size
    )
    ds = prefetch_lazy_iter_ds.__iter__()
    if warm_start:
      ds.start_prefetch()
    actual = list(ds)
    expected = list(range(1, 20, 2))
    self.assertSequenceEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_record_state_interval',
          warm_start=False,
      ),
      dict(
          testcase_name='continuous_state_recording',
          warm_start=True,
      ),
  )
  def test_checkpoint(self, warm_start: bool):
    with mock.patch.object(prefetch, '_RECORD_STATE_INTERVAL_S', 0):
      ds = prefetch.ThreadPrefetchIterDataset(
          self.ds,
          prefetch_buffer_size=500,
      )
      ds_iter = ds.__iter__()
      if warm_start:
        ds_iter.start_prefetch()

      max_steps = 10
      values_without_interruption = []
      checkpoints = []
      for _ in range(max_steps):
        checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
        values_without_interruption.append(next(ds_iter))

      for starting_step in range(9):
        ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
        for i in range(starting_step, max_steps):
          value = next(ds_iter)
          self.assertEqual(value, values_without_interruption[i])


if __name__ == '__main__':
  absltest.main()
