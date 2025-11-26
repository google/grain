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
from concurrent import futures
import dataclasses
import logging as std_logging
import os
import sys
import time
from typing import TypeVar
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch
from grain._src.python.dataset.transformations import process_prefetch
import numpy as np


_T = TypeVar('_T')


def _process_prefetch_worker_init_fn():
  log_formatter = std_logging.Formatter('[Worker 0 out of 1] %(message)s')
  logging.get_absl_handler().setFormatter(log_formatter)


@dataclasses.dataclass(frozen=True)
class FilterKeepingOddElementsOnly(transforms.Filter):

  def filter(self, element: int) -> bool:
    return bool(element % 2)


class ProcessPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds = (
        dataset.MapDataset.range(20)
        .to_iter_dataset()
        .filter(FilterKeepingOddElementsOnly())
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='process',
          buffer_size=1,
          warm_start=True,
      ),
      dict(
          testcase_name='process_large_buffer',
          buffer_size=20,
          warm_start=False,
      ),
      dict(
          testcase_name='process_huge_buffer',
          buffer_size=200,
          warm_start=True,
      ),
  )
  def test_prefetch_data(self, buffer_size: int, warm_start: bool):
    prefetch_iter_ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds, buffer_size=buffer_size
    )
    ds = prefetch_iter_ds.__iter__()
    if warm_start:
      ds.start_prefetch()
    actual = list(ds)
    expected = list(range(1, 20, 2))
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters([False, True])
  def test_checkpoint(self, warm_start: bool):
    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds,
        buffer_size=5,
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

  def test_set_state_does_not_restart_process(self):
    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds.map(lambda i: (i, os.getpid())),
        buffer_size=2,
    )
    ds_iter = ds.__iter__()
    # Read 5 elements and check that the process ID is the same.
    pids1 = [next(ds_iter)[1] for _ in range(5)]
    self.assertLen(set(pids1), 1)
    self.assertNotEqual(pids1[0], os.getpid())
    # Checkpoint, advance, and restore.
    checkpoint = ds_iter.get_state()
    next(ds_iter)  # Advance iterator.
    ds_iter.set_state(checkpoint)
    # Read 5 more elements and check that the process ID is still the same.
    pids2 = [next(ds_iter)[1] for _ in range(5)]
    self.assertLen(set(pids2), 1)
    self.assertEqual(pids1[0], pids2[0])

  def test_set_state_on_fresh_iterator(self):
    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds,
        buffer_size=2,
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
    event = mp.get_context('spawn').Event()

    def f(x):
      event.set()
      return x

    ds = dataset.MapDataset.source([1, 2, 3]).map(f).to_iter_dataset()
    ds = process_prefetch.ProcessPrefetchIterDataset(ds, buffer_size=10)
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
    ds = process_prefetch.ProcessPrefetchIterDataset(parent, buffer_size=1)
    ds_iter = ds.__iter__()
    ds_iter.set_state({'worker_state': {'test': 2}, 'iterations_to_skip': 0})
    self.assertEqual(
        ds_iter.get_state(),
        {'worker_state': {'test': 2}, 'iterations_to_skip': 0},
    )
    with self.assertRaisesRegex(
        ValueError, '`buffer_size` must be greater than 0'
    ):
      process_prefetch.ProcessPrefetchIterDataset(self.ds, buffer_size=0)
    with self.assertRaisesRegex(
        ValueError, '`buffer_size` must be greater than 0'
    ):
      process_prefetch.ProcessPrefetchIterDataset(self.ds, buffer_size=-1)

  def test_does_not_hang_after_stop_iteration(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(2).to_iter_dataset()
    ds = process_prefetch.ProcessPrefetchIterDataset(ds, buffer_size=10)
    it = ds.__iter__()
    self.assertLen(list(it), 6)
    self.assertEmpty(list(it))

  def test_worker_init_fn(self):
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
    ds = process_prefetch.ProcessPrefetchIterDataset(
        ds,
        buffer_size=1,
        worker_init_fn=_process_prefetch_worker_init_fn,
    )
    self.assertEqual(
        list(ds),
        [
            '[Worker 0 out of 1] processing element 0',
            '[Worker 0 out of 1] processing element 1',
        ],
    )

  def test_propagates_transform_error(self):
    error_msg = 'I shall fail!'

    def failing_transform(element):
      del element
      raise ValueError(error_msg)

    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds.map(failing_transform),
        buffer_size=1,
    )
    with self.assertRaisesRegex(Exception, error_msg):
      list(ds)

  def test_reports_unpicklable_transform(self):
    class UnpicklableObject:

      def __getstate__(self):
        raise ValueError('UnpicklableObject is not picklable')

    local_state = UnpicklableObject()

    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds.map(lambda _: 1 if local_state is None else 2),
        buffer_size=1,
    )
    with self.assertRaisesRegex(
        ValueError, 'UnpicklableObject is not picklable'
    ):
      list(ds)

  def test_fails_with_nested_prefetch(self):
    ds1 = process_prefetch.ProcessPrefetchIterDataset(self.ds, buffer_size=1)
    with self.assertRaisesRegex(
        ValueError,
        'Nesting prefetching with processes is not allowed',
    ):
      process_prefetch.ProcessPrefetchIterDataset(ds1, buffer_size=1)

    ds2 = prefetch.MultiprocessPrefetchIterDataset(
        self.ds, options.MultiprocessingOptions(num_workers=1)
    )
    with self.assertRaisesRegex(
        ValueError,
        'Nesting prefetching with processes is not allowed',
    ):
      process_prefetch.ProcessPrefetchIterDataset(ds2, buffer_size=1)

  def test_reports_worker_crash(self):
    def failing_transform(element):
      del element
      sys.exit(123)

    ds = process_prefetch.ProcessPrefetchIterDataset(
        self.ds.map(failing_transform),
        buffer_size=1,
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r'Worker process was terminated unexpectedly with exit code 123.*',
    ):
      list(ds)

  def test_options_before_prefetch(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(1000)
    ds = ds.to_iter_dataset()
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    ds = process_prefetch.ProcessPrefetchIterDataset(ds, buffer_size=1)
    ds = ds.filter(lambda x: x > 2)
    with self.assertRaises(Exception):
      list(ds)

  def test_options_after_prefetch(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(1000)
    ds = ds.filter(lambda x: x > 2)
    ds = ds.to_iter_dataset()
    ds = process_prefetch.ProcessPrefetchIterDataset(ds, buffer_size=1)
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    with self.assertRaises(Exception):
      list(ds)


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


class MultiprocessingPrefetchTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    ds = dataset.MapDataset.range(20)
    self.iter_ds = ds.to_iter_dataset().filter(FilterKeepingOddElementsOnly())

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
    prefetch_lazy_iter_ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds,
        num_workers=num_workers,
        buffer_size=per_worker_buffer_size,
    )
    actual = list(prefetch_lazy_iter_ds)
    expected = list(range(1, 20, 2))
    self.assertSequenceEqual(actual, expected)

  def test_prefetch_size_zero_data(self):
    ds = dataset.MapDataset.source(
        [np.zeros(shape=(0,), dtype=np.int64)]
    ).repeat(3)
    iter_ds = ds.to_iter_dataset()
    prefetch_lazy_iter_ds = process_prefetch.multiprocess_prefetch(
        iter_ds,
        num_workers=1,
    )
    actual = list(prefetch_lazy_iter_ds)
    expected = [np.zeros(shape=(0,), dtype=np.int64)] * 3
    self.assertLen(actual, 3)
    self.assertLen(expected, 3)
    for i in range(3):
      np.testing.assert_array_equal(actual[i], expected[i])

  @parameterized.product(
      (
          dict(num_workers=0),
          dict(num_workers=1),
          dict(num_workers=10),
      ),
      step_index=[0, 3, 8],
  )
  def test_checkpoint(self, num_workers: int, step_index: int):
    ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds,
        num_workers=num_workers,
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
    ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds,
        num_workers=2,
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

  def test_works_with_iter_source_single_worker(self):
    # Even though a pure IterDataset cannot be sliced, we should still be able
    # to multiprocess-prefetch it with a single worker, since that doesn't
    # require any slicing.
    ds = process_prefetch.multiprocess_prefetch(
        RepeatedIntSourceIterDataset().map(lambda x: x + 1),
        num_workers=1,
    )
    ds_iter = iter(ds)
    self.assertEqual(next(ds_iter), 2)

  def test_fails_with_iter_source_multiple_workers(self):
    with self.assertRaisesRegex(
        ValueError,
        'Cannot slice `IterDataset` source.',
    ):
      process_prefetch.multiprocess_prefetch(
          RepeatedIntSourceIterDataset().map(lambda x: x + 1),
          num_workers=2,
      )

  def test_propagates_transform_error(self):
    error_msg = 'I shall fail!'

    def failing_transform(element):
      del element
      raise ValueError(error_msg)

    ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds.map(failing_transform),
        num_workers=1,
    )
    with self.assertRaisesRegex(Exception, error_msg):
      list(ds)

  def test_reports_worker_crash(self):
    def failing_transform(element):
      del element
      sys.exit(123)

    ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds.map(failing_transform),
        num_workers=1,
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

    ds = process_prefetch.multiprocess_prefetch(
        self.iter_ds.map(lambda _: 1 if local_state is None else 2),
        num_workers=1,
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
    iter_ds = process_prefetch.multiprocess_prefetch(iter_ds, num_workers=1)

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
    iter_ds = process_prefetch.multiprocess_prefetch(iter_ds, num_workers=1)

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
    ds = ds.to_iter_dataset()
    ds = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=num_workers,
        buffer_size=per_worker_buffer_size,
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
    ds = process_prefetch.multiprocess_prefetch(ds, num_workers=1)
    it = ds.__iter__()
    it.start_prefetch()
    time.sleep(sleep_s)
    del it

  def test_prefetch_with_random_map(self):
    ds = dataset.MapDataset.source([0]).repeat(100).to_iter_dataset()
    ds = ds.random_map(lambda x, rng: x + rng.integers(sys.maxsize), seed=42)
    ds = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=5,
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
      ds = process_prefetch.multiprocess_prefetch(ds, num_workers=1)
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
    ds = process_prefetch.multiprocess_prefetch(ds, num_workers=1)
    ds = ds.filter(lambda x: x > 2)
    with self.assertRaises(Exception):
      list(ds)

  def test_multiprocess_prefetch_with_sequential_slice(self):
    ds = dataset.MapDataset.source(range(10)).to_iter_dataset()
    ds = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    self.assertEqual(list(ds), [0, 4, 7, 1, 5, 8, 2, 6, 9, 3])

  def test_multiprocess_prefetch_with_default_slice_non_sequential(self):
    ds = dataset.MapDataset.source(range(10)).to_iter_dataset()
    ds_sequential_off = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=3,
        buffer_size=1,
        sequential_slice=False,
    )
    ds_sequential_default = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=3,
        buffer_size=1,
    )
    elements_sequential_off = list(ds_sequential_off)
    elements_sequential_default = list(ds_sequential_default)
    self.assertEqual(
        elements_sequential_off,
        elements_sequential_default,
    )
    self.assertEqual(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        elements_sequential_default,
    )

  def test_multiprocess_prefetch_sequential_slice_order_from_source(self):
    ds = dataset.MapDataset.source(range(10)).to_iter_dataset()
    ds_sequential_on = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    elements_sequential_on = list(ds_sequential_on)
    self.assertEqual([0, 4, 7, 1, 5, 8, 2, 6, 9, 3], elements_sequential_on)

  def test_multiprocess_prefetch_sequential_slice_order_from_range(self):
    ds_range = dataset.MapDataset.range(10).to_iter_dataset()
    ds_range_sequential_on = process_prefetch.multiprocess_prefetch(
        ds_range,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    elements_range_sequential_on = list(ds_range_sequential_on)
    self.assertEqual(
        [0, 4, 7, 1, 5, 8, 2, 6, 9, 3],
        elements_range_sequential_on,
    )

  def test_multiprocess_prefetch_sequential_slice_order_from_range_slice(self):
    ds_range = dataset.MapDataset.range(
        start=2, stop=21, step=3
    ).to_iter_dataset()
    ds_range_sequential_on = process_prefetch.multiprocess_prefetch(
        ds_range,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    elements_range_sequential_on = list(ds_range_sequential_on)
    self.assertEqual(
        [2, 11, 17, 5, 14, 20, 8],
        elements_range_sequential_on,
    )

  def test_multiprocess_prefetch_sequential_slice_order_same(self):
    ds_source = dataset.MapDataset.source(range(10)).to_iter_dataset()
    ds_range = dataset.MapDataset.range(10).to_iter_dataset()
    ds_source_mp = process_prefetch.multiprocess_prefetch(
        ds_source,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    ds_range_mp = process_prefetch.multiprocess_prefetch(
        ds_range,
        num_workers=3,
        buffer_size=1,
        sequential_slice=True,
    )
    elements_source = list(ds_source_mp)
    elements_range = list(ds_range_mp)
    self.assertEqual(elements_source, elements_range)

  def test_options_after_prefetch(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(1000)
    ds = ds.filter(lambda x: x > 2)
    ds = ds.to_iter_dataset()
    ds = process_prefetch.multiprocess_prefetch(ds, num_workers=1)
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
    ds = process_prefetch.multiprocess_prefetch(
        ds,
        num_workers=2,
        worker_init_fn=set_worker_index_and_count,
    )
    self.assertEqual(
        list(ds),
        [
            '[Worker 0 out of 2] processing element 0',
            '[Worker 1 out of 2] processing element 1',
        ],
    )


if __name__ == '__main__':
  absltest.main()
