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
import dataclasses
import platform
import sys
import threading
import time
from typing import TypeVar, cast
from unittest import mock

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
    self.assertEmpty(ds_iter._buffer)
    _ = next(ds_iter)
    self.assertLen(ds_iter._buffer, prefetch_buffer_size)
    _ = [next(ds_iter) for _ in range(14)]
    self.assertLen(
        ds_iter._buffer, len(self.range_ds) - prefetch_buffer_size
    )  # iterated through 15 elements so far
    _ = [next(ds_iter) for _ in range(5)]
    self.assertEmpty(ds_iter._buffer)  # iterated through all elements

  def test_set_prefetch_buffer_size_0_to_positive(self):
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions(prefetch_buffer_size=0)
    )
    ds_iter = iter(prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)

    # With prefetch_buffer_size=0, executor is not created.
    self.assertFalse(hasattr(ds_iter, '_executor'))
    self.assertEqual(ds_iter._prefetch_buffer_size, 0)
    self.assertEqual(next(ds_iter), 0)

    # Setting prefetch_buffer_size to 2.
    ds_iter.set_prefetch_buffer_size(2)
    self.assertEqual(ds_iter._prefetch_buffer_size, 2)
    self.assertEqual(next(ds_iter), 1)
    self.assertTrue(hasattr(ds_iter, '_executor'))
    self.assertLen(ds_iter._buffer, 2)
    self.assertEqual(next(ds_iter), 2)
    self.assertLen(ds_iter._buffer, 2)

  def test_set_prefetch_buffer_size_positive_to_0(self):
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions(prefetch_buffer_size=2)
    )
    ds_iter = iter(prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)

    self.assertEqual(ds_iter._prefetch_buffer_size, 2)
    self.assertEqual(next(ds_iter), 0)
    self.assertLen(ds_iter._buffer, 2)

    # Setting prefetch_buffer_size to 0.
    ds_iter.set_prefetch_buffer_size(0)
    self.assertEqual(ds_iter._prefetch_buffer_size, 0)
    # Should consume buffer first.
    self.assertEqual(next(ds_iter), 1)
    self.assertLen(ds_iter._buffer, 1)
    self.assertEqual(next(ds_iter), 2)
    self.assertEmpty(ds_iter._buffer)
    # Buffer empty, should read without prefetching.
    self.assertEqual(next(ds_iter), 3)
    self.assertEmpty(ds_iter._buffer)

  def test_set_prefetch_buffer_size_increase(self):
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions(prefetch_buffer_size=1)
    )
    ds_iter = iter(prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)

    self.assertEqual(ds_iter._prefetch_buffer_size, 1)
    self.assertEqual(next(ds_iter), 0)
    self.assertLen(ds_iter._buffer, 1)

    # Setting prefetch_buffer_size to 2.
    ds_iter.set_prefetch_buffer_size(2)
    self.assertEqual(ds_iter._prefetch_buffer_size, 2)
    self.assertEqual(next(ds_iter), 1)
    self.assertLen(ds_iter._buffer, 2)
    self.assertEqual(next(ds_iter), 2)
    self.assertLen(ds_iter._buffer, 2)

  def test_set_prefetch_buffer_size_decrease(self):
    prefetch_lazy_iter_ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions(prefetch_buffer_size=2)
    )
    ds_iter = iter(prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)

    self.assertEqual(ds_iter._prefetch_buffer_size, 2)
    self.assertEqual(next(ds_iter), 0)
    self.assertLen(ds_iter._buffer, 2)

    # Setting prefetch_buffer_size to 1.
    ds_iter.set_prefetch_buffer_size(1)
    self.assertEqual(ds_iter._prefetch_buffer_size, 1)
    self.assertEqual(next(ds_iter), 1)
    self.assertLen(ds_iter._buffer, 1)
    self.assertEqual(next(ds_iter), 2)
    self.assertLen(ds_iter._buffer, 1)

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
    ds = dataset.MapDataset.range(0, 1000).map(lambda x: x).to_iter_dataset()
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

  def test_set_num_threads_decrease_threads(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)
    self.assertEqual(ds_iter._num_threads, options.ReadOptions().num_threads)
    self.assertEqual(
        ds_iter._executor._max_workers, options.ReadOptions().num_threads
    )
    self.assertEqual([next(ds_iter) for _ in range(5)], list(range(5)))

    # Decrease threads
    ds_iter.set_num_threads(5)
    self.assertEqual(ds_iter._num_threads, 5)
    self.assertEqual(ds_iter._executor._max_workers, 5)
    self.assertEqual([next(ds_iter) for _ in range(15)], list(range(5, 20)))

  def test_set_num_threads_increase_threads(self):
    ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions(num_threads=5)
    )
    ds_iter = iter(ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)
    self.assertEqual(ds_iter._num_threads, 5)
    self.assertEqual(ds_iter._executor._max_workers, 5)
    self.assertEqual([next(ds_iter) for _ in range(5)], list(range(5)))

    # Increase threads
    ds_iter.set_num_threads(10)
    self.assertEqual(ds_iter._num_threads, 10)
    self.assertEqual(ds_iter._executor._max_workers, 10)
    self.assertEqual([next(ds_iter) for _ in range(15)], list(range(5, 20)))

  def test_set_num_threads_decrease_to_zero(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)
    self.assertEqual(ds_iter._num_threads, options.ReadOptions().num_threads)
    self.assertEqual(
        ds_iter._executor._max_workers, options.ReadOptions().num_threads
    )
    self.assertEqual([next(ds_iter) for _ in range(5)], list(range(5)))
    # Decrease threads to 0
    ds_iter.set_num_threads(0)
    self.assertEqual(ds_iter._num_threads, 0)
    self.assertFalse(hasattr(ds_iter, '_executor'))
    self.assertEqual([next(ds_iter) for _ in range(15)], list(range(5, 20)))

  def test_set_num_threads_increase_from_zero(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, prefetch.PrefetchDatasetIterator)
    ds_iter = cast(prefetch.PrefetchDatasetIterator, ds_iter)
    self.assertEqual([next(ds_iter) for _ in range(5)], list(range(5)))
    ds_iter.set_num_threads(0)
    self.assertEqual(ds_iter._num_threads, 0)
    self.assertFalse(hasattr(ds_iter, '_executor'))
    self.assertEqual([next(ds_iter) for _ in range(5)], list(range(5, 10)))

    # Increase threads from 0
    ds_iter.set_num_threads(5)
    self.assertEqual(ds_iter._num_threads, 5)
    self.assertEqual(ds_iter._executor._max_workers, 5)
    self.assertEqual([next(ds_iter) for _ in range(10)], list(range(10, 20)))

  @parameterized.product(
      start_prefetch_calls=[1, 10],
      num_threads=[0, 16],
  )
  def test_start_prefetch(
      self,
      start_prefetch_calls: int,
      num_threads: int,
  ):
    ds = dataset.MapDataset.range(10)
    ds = ds.map(lambda x: x)
    ds = prefetch.PrefetchIterDataset(
        ds, read_options=options.ReadOptions(num_threads)
    )

    it = ds.__iter__()
    for _ in range(start_prefetch_calls):
      it.start_prefetch()
    # Check that the buffer was filled before we start processing elements.
    if num_threads > 0:
      self.assertNotEmpty(it._buffer)  # pytype: disable=attribute-error

    self.assertEqual(list(it), list(range(10)))

  @parameterized.parameters(True, False)
  def test_stats_are_initialized_in_a_single_thread(self, start_prefetch: bool):
    stats_init_threads = set()

    class FilterMapDataset(filter_lazy_dataset.FilterMapDataset):

      def _initialize_stats(self, *args, **kwargs):
        stats_init_threads.add(threading.get_ident())
        return super()._initialize_stats(*args, **kwargs)

    ds = FilterMapDataset(dataset.MapDataset.range(10), lambda x: x > 2)
    ds = prefetch.PrefetchIterDataset(ds, read_options=options.ReadOptions())
    it = ds.__iter__()
    if start_prefetch:
      it.start_prefetch()
    _ = list(it)
    self.assertEqual(stats_init_threads, {threading.get_ident()})

  def test_element_spec(self):
    ds = prefetch.PrefetchIterDataset(
        self.range_ds, read_options=options.ReadOptions()
    )
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.shape, ())
    self.assertEqual(spec.dtype, np.int64)

  def test_get_next_index(self):
    ds_iter = self.prefetch_lazy_iter_ds.__iter__()
    for i in range(20):
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      _ = next(ds_iter)

  def test_set_next_index(self):
    ds_iter = self.prefetch_lazy_iter_ds.__iter__()
    for i in reversed(range(20)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), i)


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

  def test_does_not_hang_after_stop_iteration(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(100).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    it = ds.__iter__()
    _ = list(it)
    self.assertEmpty(list(it))

  def test_nonnative_iterator(self):

    class TestIterator:

      def __init__(self):
        self._counter = 0

      def __iter__(self):
        return self

      def __next__(self) -> int:
        self._counter += 1
        if self._counter > 10:
          raise StopIteration
        return self._counter

      def get_state(self):
        return {'counter': self._counter}

      def set_state(self, state):
        self._counter = state['counter']

    test_iterator = TestIterator()
    it = prefetch.ThreadPrefetchDatasetIterator(
        test_iterator, prefetch_buffer_size=10
    )
    elements = []
    checkpoint_step = 5
    for _ in range(checkpoint_step):
      elements.append(next(it))
    checkpoint = it.get_state()
    elements.extend(it)
    self.assertEqual(elements, list(range(1, 11)))
    it.set_state(checkpoint)
    self.assertEqual(list(it), elements[checkpoint_step:])

  def test_no_mem_leak(self):
    ds = (
        dataset.MapDataset.range(1000)
        .repeat()
        .map(lambda x: x * np.ones((1000, 1000), dtype=np.int64))
        .to_iter_dataset(options.ReadOptions(prefetch_buffer_size=0))
    )
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    # If buffered elements are not cleaned up when the iterator is gc'ed, this
    # test will OOM.
    for _ in range(1000):
      it = ds.__iter__()
      for _ in range(5):
        _ = next(it)

  @parameterized.parameters([True, False])
  def test_no_mem_leak_with_double_prefetch(self, close: bool):
    ds = (
        dataset.MapDataset.range(1000)
        .repeat()
        .map(lambda x: x * np.ones((1000, 1000), dtype=np.int64))
        .to_iter_dataset(options.ReadOptions(prefetch_buffer_size=0))
    )
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    ds = ds.map(lambda x: x + 1)
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    # If buffered elements are not cleaned up when the iterator is gc'ed, this
    # test will OOM.
    for _ in range(1000):
      it = ds.__iter__()
      for _ in range(5):
        _ = next(it)
      if close:
        it.close()  # pytype: disable=attribute-error

  @absltest.skipIf(platform.system() == 'Darwin', 'Fails on macos-14 runner.')
  @parameterized.parameters([True, False])
  def test_early_break_continues_prefetching(self, close: bool):
    count = 0
    count_lock = threading.Lock()

    class SlowCountingSource:

      def __len__(self):
        return 16

      def __getitem__(self, index):
        nonlocal count
        time.sleep(0.1)
        with count_lock:
          count += 1
        return index

    read_options = options.ReadOptions(num_threads=2)
    ds = dataset.MapDataset.source(SlowCountingSource()).to_iter_dataset(
        read_options
    )
    iterator = ds.__iter__()

    assert count == 0
    if close:
      next(iterator)
      self.assertGreater(count, 0)
      iterator.close()
      time.sleep(1)
      self.assertLess(count, 8)
    else:
      next(iterator)
      self.assertGreater(count, 0)
      time.sleep(1)
      self.assertGreater(count, 8)

  def test_element_spec(self):
    ds = dataset.MapDataset.range(2).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=1)
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())

  def test_get_next_index(self):
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    ds_iter = ds.__iter__()
    for i in range(20):
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      _ = next(ds_iter)

  def test_set_next_index(self):
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    ds_iter = ds.__iter__()
    for i in reversed(range(20)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), i)

  def test_set_next_index_get_state(self):
    # If `get_state` is called after `set_next_index` and before the next
    # `__next__` call, the iterator should call `get_state` on the parent
    # iterator in its `get_state` implementation, not in the `set_next_index`
    # implementation.
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = prefetch.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=10)
    get_state_counter = mock.Mock()
    get_state = prefetch.PrefetchDatasetIterator.get_state

    main_thread_id = threading.get_ident()

    def new_get_state(self):
      if threading.get_ident() == main_thread_id:
        # Only count `get_state` calls from the main thread.
        get_state_counter()
      get_state(self)

    with mock.patch.object(
        prefetch.PrefetchDatasetIterator, 'get_state', new_get_state
    ):
      ds_iter = ds.__iter__()
      next(ds_iter)
      get_state_count = get_state_counter.call_count
      dataset.set_next_index(ds_iter, 5)
      self.assertEqual(get_state_counter.call_count - get_state_count, 0)
      ds_iter.get_state()
      self.assertEqual(get_state_counter.call_count - get_state_count, 1)


class _MpContextCheckIterDataset(dataset.IterDataset[_T]):

  def __iter__(self) -> dataset.DatasetIterator[_T]:
    return _MpContextCheckIterator(self._parent.__iter__())


class _MpContextCheckIterator(dataset.DatasetIterator[_T]):

  def __next__(self) -> tuple[_T, base.MultiprocessingContext]:
    element = next(self._parent)
    return (element, self._ctx.mp_context)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)


class MultithreadPrefetchIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds = dataset.MapDataset.range(20).to_iter_dataset()

  @parameterized.named_parameters(
      dict(
          testcase_name='no_prefetch',
          num_workers=0,
          per_worker_buffer_size=0,
      ),
      dict(
          testcase_name='thread',
          num_workers=1,
          per_worker_buffer_size=1,
      ),
      dict(
          testcase_name='2_threads_large_buffer',
          num_workers=2,
          per_worker_buffer_size=20,
      ),
      dict(
          testcase_name='4_threads_huge_buffer',
          num_workers=4,
          per_worker_buffer_size=200,
      ),
  )
  def test_prefetch_data(self, num_workers: int, per_worker_buffer_size: int):
    prefetch_lazy_iter_ds = prefetch.multithread_prefetch(
        self.ds,
        num_threads=num_workers,
        buffer_size=per_worker_buffer_size,
    )
    ds_iter = prefetch_lazy_iter_ds.__iter__()
    if num_workers > 0:
      ds_iter.start_prefetch()
    actual = list(ds_iter)
    expected = list(range(20))
    self.assertSequenceEqual(actual, expected)

  def test_checkpoint(self):
    ds = prefetch.multithread_prefetch(
        self.ds,
        num_threads=2,
        buffer_size=5,
    )
    ds_iter = ds.__iter__()
    ds_iter.start_prefetch()

    max_steps = 20
    values_without_interruption = []
    checkpoints = []
    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 5, 13, 19]:
      ds_iter.set_state(checkpoints[starting_step])
      ds_iter.start_prefetch()
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        print(value)
        self.assertEqual(value, values_without_interruption[i])

  def test_set_state_on_fresh_iterator(self):
    ds = prefetch.multithread_prefetch(
        self.ds,
        num_threads=2,
        buffer_size=2,
    )
    ds_iter = ds.__iter__()
    ds_iter.start_prefetch()

    max_steps = 20
    values_without_interruption = []
    checkpoints = []
    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 5, 13, 19]:
      ds_iter = ds.__iter__()
      ds_iter.set_state(checkpoints[starting_step])
      ds_iter.start_prefetch()
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])

  def test_does_not_hang_after_stop_iteration(self):
    ds = dataset.MapDataset.source([1, 2, 3]).repeat(100).to_iter_dataset()
    ds = prefetch.multithread_prefetch(
        ds,
        num_threads=2,
        buffer_size=10,
    )
    it = ds.__iter__()
    it.start_prefetch()

  def test_mp_context_is_set_correctly(self):
    num_workers = 4
    ds = dataset.MapDataset.range(20).to_iter_dataset()
    ds = _MpContextCheckIterDataset(ds)
    ds = ds.map(lambda x: x)
    ds = prefetch.multithread_prefetch(
        ds,
        num_threads=num_workers,
        buffer_size=1,
    )

    results = list(ds)
    self.assertLen(results, 20)

    # Check that elements are interleaved correctly.
    elements = [r[0] for r in results]
    self.assertEqual(elements, list(range(20)))

    # Check mp_context.
    for i, (_, context) in enumerate(results):
      self.assertEqual(context.process_index, i % num_workers)
      self.assertEqual(context.process_count, num_workers)


if __name__ == '__main__':
  absltest.main()
