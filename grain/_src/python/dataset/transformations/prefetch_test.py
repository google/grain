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
from concurrent import futures
import dataclasses
import logging as std_logging
import sys
import threading
import time
from typing import TypeVar, cast
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import filter as filter_lazy_dataset
from grain._src.python.dataset.transformations import prefetch
import numpy as np


_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True)
class FilterKeepingOddElementsOnly(transforms.Filter):

  def filter(self, element: int) -> bool:
    return bool(element % 2)


@dataclasses.dataclass(frozen=True)
class FilterAllElements(transforms.Filter):

  def filter(self, element: int):
    return False


class RepeatedIntSourceIterDataset(dataset.IterDataset[int]):

  def __iter__(self) -> dataset.DatasetIterator[int]:
    return RepeatedIntSourceDatasetIterator()


class RepeatedIntSourceDatasetIterator(dataset.DatasetIterator[int]):

  def __iter__(self) -> dataset.DatasetIterator[int]:
    return self

  def __next__(self) -> int:
    return 1

  def set_state(self, state):
    pass

  def get_state(self):
    return {}


class PrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(20)
    self.filtered_range_ds = self.range_ds.filter(
        FilterKeepingOddElementsOnly()
    )
    self.prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions()
    )
    filter_lazy_dataset._WARN_FILTERED_INTERVAL_SEC = 0.0

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

  def test_filter_all_elements_warns(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .filter(FilterAllElements())
        .to_iter_dataset()
    )
    with self.assertLogs(level='WARNING') as logs:
      _ = list(ds)
    logs = logs[0][0].message
    self.assertRegex(
        logs,
        r'Transformation'
        r' PrefetchDatasetIterator\(read_options=ReadOptions\(num_threads=16,'
        r' prefetch_buffer_size=500\), allow_nones=False\)'
        r' skipped 100.00 \% of the last seen 1000 elements.',
    )

  def test_filter_all_elements_raises(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .filter(FilterAllElements())
        .to_iter_dataset()
    )
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.9)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    with self.assertRaisesRegex(
        ValueError,
        r'Transformation'
        r' PrefetchDatasetIterator\(read_options=ReadOptions\(num_threads=16,'
        r' prefetch_buffer_size=500\), allow_nones=False\)'
        r' skipped 100.00 \% of the last seen 1000 elements.',
    ):
      _ = list(ds)

  def test_filter_all_elements_doesnt_raise_with_allow_nones(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .filter(FilterAllElements())
        .to_iter_dataset(allow_nones=True)
    )
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.9)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    self.assertEqual(list(ds), [None] * 1000)

  def test_iterator_has_no_reference_cycle(self):
    ds = (
        dataset.MapDataset.range(0, 1000)
        .map(lambda x: x)
        .to_iter_dataset()
    )
    ds_iter = iter(ds)
    # Here, we check that iterating over the data does not create new references
    # to the iterator. One common scenario when this could happen is if the
    # iterator is passed to the prefetching threads. This is a problem because
    # it delays garbage collection of all objects referred to by the iterator,
    # including the buffered data.
    ref_count_before = sys.getrefcount(ds_iter)
    for _ in range(1000):
      next(ds_iter)
      self.assertEqual(sys.getrefcount(ds_iter), ref_count_before)


class MultiprocessPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    ds = dataset.MapDataset.range(20)
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    self.iter_ds = ds.filter(FilterKeepingOddElementsOnly())

  @parameterized.named_parameters(
      dict(
          testcase_name='0_workers',
          num_workers=0,
          per_worker_buffer_size=1,
      ),
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

  def test_prefetch_size_zero_data(self):
    ds = dataset.MapDataset.source(
        [np.zeros(shape=(0,), dtype=np.int64)]
    ).repeat(3)
    iter_ds = ds.to_iter_dataset()
    prefetch_lazy_iter_ds = prefetch.MultiprocessPrefetchIterDataset(
        iter_ds,
        options.MultiprocessingOptions(num_workers=1),
    )
    actual = list(prefetch_lazy_iter_ds)
    expected = [np.zeros(shape=(0,), dtype=np.int64)] * 3
    self.assertLen(actual, 3)
    self.assertLen(expected, 3)
    for i in range(3):
      np.testing.assert_array_equal(actual[i], expected[i])

  @parameterized.product(
      (
          dict(
              num_workers=0,
              record_state_interval=prefetch._RECORD_STATE_INTERVAL_S,
          ),
          dict(
              num_workers=1,
              record_state_interval=prefetch._RECORD_STATE_INTERVAL_S,
          ),
          dict(
              num_workers=10,
              record_state_interval=prefetch._RECORD_STATE_INTERVAL_S,
          ),
          dict(
              num_workers=10,
              record_state_interval=0,
          ),
      ),
      step_index=[0, 3, 8],
  )
  def test_checkpoint(
      self, num_workers: int, record_state_interval: int, step_index: int
  ):
    with mock.patch.object(
        prefetch, '_RECORD_STATE_INTERVAL_S', record_state_interval
    ):
      ds = prefetch.MultiprocessPrefetchIterDataset(
          self.iter_ds,
          options.MultiprocessingOptions(num_workers),
      )
      ds_iter = ds.__iter__()

      max_steps = 10
      values_without_interruption = []
      checkpoints = []
      for _ in range(max_steps):
        checkpoints.append(ds_iter.get_state())
        values_without_interruption.append(next(ds_iter))

      ds_iter.set_state(checkpoints[step_index])
      for i in range(step_index, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])

  def test_set_state_twice(self):
    with mock.patch.object(prefetch, '_RECORD_STATE_INTERVAL_S', 0):
      ds = prefetch.MultiprocessPrefetchIterDataset(
          self.iter_ds,
          options.MultiprocessingOptions(2),
      )
      ds_iter = ds.__iter__()

      max_steps = 10
      values_without_interruption = []
      checkpoints = []
      for _ in range(max_steps):
        checkpoints.append(ds_iter.get_state())
        values_without_interruption.append(next(ds_iter))

      for starting_step in [0, 3, 8]:
        ds_iter.set_state(checkpoints[starting_step])
        for i in range(starting_step, max_steps):
          value = next(ds_iter)
          self.assertEqual(value, values_without_interruption[i])

  def test_fails_with_negative_num_workers(self):
    with self.assertRaisesRegex(
        ValueError, '`num_workers` must be greater than or equal to 0'
    ):
      prefetch.MultiprocessPrefetchIterDataset(
          self.iter_ds,
          options.MultiprocessingOptions(num_workers=-1),
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

  def test_works_with_iter_source_single_worker(self):
    # Even though a pure IterDataset cannot be sliced, we should still be able
    # to multiprocess-prefetch it with a single worker, since that doesn't
    # require any slicing.
    ds = prefetch.MultiprocessPrefetchIterDataset(
        RepeatedIntSourceIterDataset().map(lambda x: x + 1),
        options.MultiprocessingOptions(num_workers=1),
    )
    ds_iter = iter(ds)
    self.assertEqual(next(ds_iter), 2)

  def test_fails_with_iter_source_multiple_workers(self):
    ds = prefetch.MultiprocessPrefetchIterDataset(
        RepeatedIntSourceIterDataset().map(lambda x: x + 1),
        options.MultiprocessingOptions(num_workers=2),
    )
    ds_iter = iter(ds)

    with self.assertRaisesRegex(
        Exception,
        'Cannot slice `IterDataset` source.',
    ):
      next(ds_iter)

  def test_propagates_transform_error(self):
    error_msg = 'I shall fail!'

    def failing_transform(element):
      del element
      raise ValueError(error_msg)

    ds = prefetch.MultiprocessPrefetchIterDataset(
        self.iter_ds.map(failing_transform),
        options.MultiprocessingOptions(num_workers=1),
    )
    with self.assertRaisesRegex(Exception, error_msg):
      list(ds)

  def test_reports_worker_crash(self):
    def failing_transform(element):
      del element
      sys.exit(123)

    ds = prefetch.MultiprocessPrefetchIterDataset(
        self.iter_ds.map(failing_transform),
        options.MultiprocessingOptions(num_workers=1),
    )
    with self.assertRaisesRegex(
        RuntimeError, 'was terminated unexpectedly with exit code 123'
    ):
      list(ds)

  def test_reports_unpicklable_transform(self):
    class UnpicklableObject:

      def __getstate__(self):
        raise ValueError('UnpicklableObject is not picklable')

    local_state = UnpicklableObject()

    ds = prefetch.MultiprocessPrefetchIterDataset(
        self.iter_ds.map(lambda _: 1 if local_state is None else 2),
        options.MultiprocessingOptions(num_workers=1),
    )
    with self.assertRaisesRegex(
        ValueError, 'UnpicklableObject is not picklable'
    ) as context_manager:
      list(ds)

    if sys.version_info >= (3, 11):
      self.assertRegex(
          ''.join(context_manager.exception.__notes__),
          r'Dataset: MapIterDataset.* cannot be pickled!',
      )

  def test_reports_first_unpicklable_dataset_when_with_multiple_parents(self):
    class UnpicklableObject:

      def __getstate__(self):
        raise ValueError('UnpicklableObject is not picklable')

    local_unpicklable_obj = UnpicklableObject()

    class LeftTransform(transforms.MapTransform):

      def map(self, x):
        return x if local_unpicklable_obj else x

    class RightTransform(transforms.MapTransform):

      def map(self, x):
        return x if local_unpicklable_obj else x

    ds_left = dataset.MapDataset.range(0, 10)
    ds_left = ds_left.map(LeftTransform())
    ds_right = dataset.MapDataset.range(10, 20)
    ds_right = ds_right.map(RightTransform())

    ds = dataset.MapDataset.mix([ds_left, ds_right], [1.0, 1.0])

    iter_ds = ds.to_iter_dataset(
        read_options=options.ReadOptions(prefetch_buffer_size=0)
    )
    iter_ds = iter_ds.mp_prefetch()

    with self.assertRaisesRegex(
        ValueError,
        r'UnpicklableObject is not picklable',
    ) as context_manager:
      list(iter_ds)

    if sys.version_info >= (3, 11):
      self.assertRegex(
          ''.join(context_manager.exception.__notes__),
          r'Dataset: MapMapDataset\(transform=LeftTransform\) cannot be'
          r' pickled!',
      )

  def test_reports_unpicklable_issue_when_only_one_parent_unpicklable(self):
    class UnpicklableObject:

      def __getstate__(self):
        raise ValueError('UnpicklableObject is not picklable')

    class PickleableTransform(transforms.MapTransform):

      def map(self, x):
        return x

    local_unpicklable_obj = UnpicklableObject()

    class RightTransform(transforms.MapTransform):

      def map(self, x):
        return x if local_unpicklable_obj else x

    ds_left = dataset.MapDataset.range(0, 10)
    ds_left = ds_left.map(PickleableTransform())
    ds_right = dataset.MapDataset.range(10, 20)
    ds_right = ds_right.map(RightTransform())

    ds = dataset.MapDataset.mix([ds_left, ds_right], [1.0, 1.0])

    iter_ds = ds.to_iter_dataset(
        read_options=options.ReadOptions(prefetch_buffer_size=0)
    )
    iter_ds = iter_ds.mp_prefetch()

    with self.assertRaisesRegex(
        ValueError, 'UnpicklableObject is not picklable'
    ) as context_manager:
      list(iter_ds)

    if sys.version_info >= (3, 11):
      self.assertRegex(
          ''.join(context_manager.exception.__notes__),
          r'Dataset: MapMapDataset\(transform=RightTransform\) cannot be'
          r' pickled!',
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
    ds = ds.map(_SleepTransform())
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    ds = prefetch.MultiprocessPrefetchIterDataset(
        ds,
        options.MultiprocessingOptions(num_workers, per_worker_buffer_size),
    )

    it = ds.__iter__()
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
    # Note that we can't reliably assert the upper bound on the time it takes
    # read the dataset elements since worker startup time can vary a lot.
    if not start_prefetch_calls:
      self.assertGreater(time_to_fetch, 1)

  @parameterized.parameters(0, 0.5, 30)
  def test_prefetch_but_no_read(self, sleep_s):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat()
    ds = ds.filter(lambda x: x > 3)
    ds = ds.to_iter_dataset()
    ds = ds.mp_prefetch()
    it = ds.__iter__()
    it.start_prefetch()
    time.sleep(sleep_s)
    del it

  def test_prefetch_with_random_map(self):
    ds = dataset.MapDataset.source([0]).repeat(100).to_iter_dataset()
    ds = ds.random_map(lambda x, rng: x + rng.integers(sys.maxsize), seed=42)
    ds = prefetch.MultiprocessPrefetchIterDataset(
        ds,
        options.MultiprocessingOptions(num_workers=5),
    )
    # Make sure that sliced datasets on workers are seeded differently and thus
    # produce different random elements.
    elements = list(ds)
    distinct_elements = set(elements)
    self.assertLen(distinct_elements, len(elements))

  def test_concurrent_start_prefetch(self):
    num_iters = 10  # Can't set this much higher without Forge OOMing.

    def make_iter(i):
      ds = dataset.MapDataset.source([i])
      ds = ds.to_iter_dataset()
      ds = ds.mp_prefetch(options=options.MultiprocessingOptions(num_workers=1))
      return ds.__iter__()

    iters = [make_iter(i) for i in range(num_iters)]
    with futures.ThreadPoolExecutor(max_workers=num_iters) as executor:
      for it in iters:
        executor.submit(it.start_prefetch)
    for it in iters:
      _ = next(it)

  def test_options_before_prefetch(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(1000)
    ds = ds.to_iter_dataset()
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=1))
    ds = ds.filter(lambda x: x > 2)
    with self.assertRaises(Exception):
      list(ds)

  def test_options_after_prefetch(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(1000)
    ds = ds.filter(lambda x: x > 2)
    ds = ds.to_iter_dataset()
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=1))
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    with self.assertRaises(Exception):
      list(ds)

  def test_worker_init_fn(self):
    def set_worker_index_and_count(worker_index: int, worker_count: int):
      log_formatter = std_logging.Formatter(
          f'[Worker {worker_index} out of {worker_count}] %(message)s'
      )
      logging.get_absl_handler().setFormatter(log_formatter)

    def map_fn(x):
      # absl logging from workers is not propagated to the main process in unit
      # tests. Therefore, we manually pass the formatted log message.
      record = logging.get_absl_logger().makeRecord(
          'grain',
          logging.INFO,
          'grain_pool_test',
          123,
          f'processing element {x}',
          (),
          None,
      )
      return logging.get_absl_handler().format(record)

    ds = dataset.MapDataset.range(2).map(map_fn)
    ds = ds.to_iter_dataset()
    ds = ds.mp_prefetch(
        options.MultiprocessingOptions(num_workers=2),
        worker_init_fn=set_worker_index_and_count,
    )
    self.assertEqual(
        list(ds),
        [
            '[Worker 0 out of 2] processing element 0',
            '[Worker 1 out of 2] processing element 1',
        ],
    )


class ThreadPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds = (
        dataset.MapDataset.range(20)
        .to_iter_dataset()
        .filter(FilterKeepingOddElementsOnly())
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='no_prefetch',
          prefetch_buffer_size=0,
          warm_start=False,
      ),
      dict(
          testcase_name='no_prefetch_with_warm_start',
          prefetch_buffer_size=0,
          warm_start=True,
      ),
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

  @parameterized.parameters([False, True])
  def test_checkpoint(self, warm_start: bool):
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
      checkpoints.append(ds_iter.get_state())
      values_without_interruption.append(next(ds_iter))

    for starting_step in range(9):
      ds_iter.set_state(checkpoints[starting_step])
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])

  def test_set_state_on_fresh_iterator(self):
    ds = prefetch.ThreadPrefetchIterDataset(
        self.ds,
        prefetch_buffer_size=2,
    )
    ds_iter = ds.__iter__()

    max_steps = 10
    values_without_interruption = []
    checkpoints = []
    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())
      values_without_interruption.append(next(ds_iter))

    for starting_step in range(9):
      ds_iter = ds.__iter__()
      ds_iter.set_state(checkpoints[starting_step])
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])

  def test_get_state_doesnt_start_prefetch(self):
    event = threading.Event()

    def f(x):
      event.set()
      return x

    ds = dataset.MapDataset.source([1, 2, 3]).map(f).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    it = ds.__iter__()
    it.get_state()
    time.sleep(1)
    self.assertFalse(event.is_set())

  def test_parent_dataset_modifies_state(self):
    class TestIterator(dataset.DatasetIterator):

      def __next__(self):
        return 1

      def get_state(self):
        return {'test': 1}

      def set_state(self, state):
        pass

    class TestDataset(dataset.IterDataset):

      def __iter__(self):
        return TestIterator()

    parent = TestDataset()
    ds = prefetch.ThreadPrefetchIterDataset(parent, prefetch_buffer_size=1)
    ds_iter = ds.__iter__()
    ds_iter.set_state({'test': 2})
    self.assertEqual(ds_iter.get_state(), {'test': 1})

  def test_fails_with_negative_prefetch_buffer_size(self):
    with self.assertRaisesRegex(
        ValueError, '`prefetch_buffer_size` must be greater than or equal to 0'
    ):
      prefetch.ThreadPrefetchIterDataset(self.ds, prefetch_buffer_size=-1)

  def test_start_prefetch_with_mp_prefetch_but_no_read(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat().to_iter_dataset()
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    it = ds.__iter__()
    it.start_prefetch()
    del it

  def test_does_not_create_reference_to_itself(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(100).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    it = ds.__iter__()
    refcount_before_iteration = sys.getrefcount(it)
    _ = next(it)
    refcount_after_iteration = sys.getrefcount(it)
    self.assertEqual(refcount_before_iteration, refcount_after_iteration)


if __name__ == '__main__':
  absltest.main()
