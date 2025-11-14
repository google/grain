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
import dataclasses
import logging as std_logging
import os
import sys
import time
from typing import TypeVar

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

  def test_no_mem_leak(self):
    ds = (
        dataset.MapDataset.range(1000)
        .repeat()
        .map(lambda x: x * np.ones((1000, 1000), dtype=np.int64))
        .to_iter_dataset(options.ReadOptions(prefetch_buffer_size=0))
    )
    ds = process_prefetch.ProcessPrefetchIterDataset(ds, buffer_size=10)
    # If buffered elements are not cleaned up when the iterator is gc'ed, this
    # test will OOM.
    for _ in range(100):
      it = ds.__iter__()
      for _ in range(5):
        _ = next(it)

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


if __name__ == '__main__':
  absltest.main()
