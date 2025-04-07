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
"""Tests for GrainPool."""

from collections.abc import Iterator
import multiprocessing
import os
import signal
import sys
from typing import Any
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import config
import multiprocessing as mp
from grain._src.python import data_sources
from grain._src.python import grain_pool as gp
from grain._src.python import record
from grain._src.python.options import MultiprocessingOptions  # pylint: disable=g-importing-member


class GrainPoolTest(absltest.TestCase):

  def _join_and_assert_process_exitcode(self, process: multiprocessing.Process):
    # The process can be potentially terminated forcibly and needs a moment to
    # finalize and update the exitcode.
    process.join(timeout=gp._PROCESS_JOIN_TIMEOUT)
    self.assertIn(process.exitcode, {0, -signal.SIGTERM})

  def test_pool_with_flags_not_parsed(self):
    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 14, worker_count))

    get_element_producer_fn = GetElementProducerFn()
    # unparse the flags explicitly
    flags.FLAGS.unparse_flags()

    _ = gp.GrainPool(
        ctx=mp.get_context("spawn"),
        get_element_producer_fn=get_element_producer_fn,
        options=MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1),
    )

  def test_pool_equal_split_in_memory_data_source(self):
    in_memory_ds = data_sources.SharedMemoryDataSource(range(12))

    # 12 elements in the `in_memory_ds` are divided
    # equally among 4 processes.
    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 12, worker_count))

    get_element_producer_fn = GetElementProducerFn()

    output_elements = []
    with gp.GrainPool(
        ctx=mp.get_context("spawn"),
        get_element_producer_fn=get_element_producer_fn,
        options=MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1),
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
        # turn each element in `in_memory_ds` to their negatives.
        in_memory_ds[element.record] = -in_memory_ds[element.record]

    self.assertEqual(
        output_elements, [gp.GrainPoolElement(x, x % 4) for x in range(12)]
    )

    self.assertEqual(list(iter(in_memory_ds)), [-x for x in range(12)])

  def test_pool_equal_split(self):
    ctx = mp.get_context("spawn")

    # 16 elements divide equally among 4 processes
    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 16, worker_count))

    get_element_producer_fn = GetElementProducerFn()

    options = MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)
    output_elements = []
    with gp.GrainPool(
        ctx=ctx,
        get_element_producer_fn=get_element_producer_fn,
        options=options,
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
    expected_elements = list(
        map(
            lambda x: gp.GrainPoolElement(x, x % options.num_workers), range(16)
        )
    )
    self.assertEqual(expected_elements, output_elements)
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool.processes, options.num_workers)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool.processes:
      self._join_and_assert_process_exitcode(child_process)

  def test_pool_non_equal_split(self):
    ctx = mp.get_context("spawn")

    # 14 elements do not divide equally among 4 processes
    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 14, worker_count))

    get_element_producer_fn = GetElementProducerFn()

    options = MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)
    output_elements = []
    with gp.GrainPool(
        ctx=ctx,
        get_element_producer_fn=get_element_producer_fn,
        options=options,
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
    expected_elements = list(
        map(
            lambda x: gp.GrainPoolElement(x, x % options.num_workers), range(14)
        )
    )
    self.assertEqual(expected_elements, output_elements)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool.processes:
      self._join_and_assert_process_exitcode(child_process)

  def test_pool_kill_child(self):
    ctx = mp.get_context("spawn")

    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 14, worker_count))

    get_element_producer_fn = GetElementProducerFn()

    options = MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)
    with gp.GrainPool(
        ctx=ctx,
        get_element_producer_fn=get_element_producer_fn,
        options=options,
    ) as grain_pool:
      child_pid = grain_pool.processes[0].pid
      os.kill(child_pid, signal.SIGKILL)

    self.assertEqual(
        grain_pool.processes[0].exitcode, -1 * signal.SIGKILL.value
    )
    for child_process in grain_pool.processes[1:]:
      self._join_and_assert_process_exitcode(child_process)

  def test_pool_object_deletion(self):
    ctx = mp.get_context("spawn")

    class GetElementProducerFn(gp.GetElementProducerFn):

      def __call__(self, *, worker_index: int, worker_count: int, **kwargs):
        del self
        return iter(range(worker_index, 14, worker_count))

    get_element_producer_fn = GetElementProducerFn()

    options = MultiprocessingOptions(num_workers=4, per_worker_buffer_size=1)

    # Users should generally use the with statement, here we test if GrainPool
    # was created without the "with statement", that object deletion would
    # have child processes gracefully exited.
    grain_pool = gp.GrainPool(
        ctx=ctx,
        get_element_producer_fn=get_element_producer_fn,
        options=options,
    )

    child_processes = grain_pool.processes
    grain_pool.__del__()

    for child_process in child_processes:
      self._join_and_assert_process_exitcode(child_process)


def _make_uniform_element_producer_fn(
    last_seen_index: int = -1,
) -> gp.GetElementProducerFn:

  class _RoundrobinElementProducerFn(gp.GetElementProducerFn):

    def __call__(
        self, *, worker_index: int, worker_count: int, **kwargs
    ) -> Iterator[int]:
      del self
      yield from range(10)[last_seen_index + 1 + worker_index :: worker_count]

  return _RoundrobinElementProducerFn()


class RoundrobinRecordElementProducerFn(gp.GetElementProducerFn):

  def __call__(
      self, *, worker_index: int, worker_count: int, **kwargs
  ) -> Iterator[record.Record[int]]:
    del self
    for i in range(5)[worker_index::worker_count]:
      yield record.Record(record.RecordMetadata(i), i)


class NonUniformElementProducerFn(gp.GetElementProducerFn):

  def __call__(
      self, *, worker_index: int, worker_count: int, **kwargs
  ) -> Iterator[int]:
    del self, worker_count
    for _ in range(worker_index * 3):
      yield worker_index


class MultiProcessIteratorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="two_workers",
          get_element_producer_fn=_make_uniform_element_producer_fn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=2),
          worker_index_to_start_reading=0,
          expected=list(range(10)),
      ),
      dict(
          testcase_name="five_workers",
          get_element_producer_fn=_make_uniform_element_producer_fn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=5),
          worker_index_to_start_reading=0,
          expected=list(range(10)),
      ),
      dict(
          testcase_name="from_checkpoint",
          get_element_producer_fn=_make_uniform_element_producer_fn(5),
          multiprocessing_options=MultiprocessingOptions(num_workers=2),
          worker_index_to_start_reading=1,
          expected=[7, 6, 9, 8],
      ),
      dict(
          testcase_name="non_uniform",
          get_element_producer_fn=NonUniformElementProducerFn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=3),
          worker_index_to_start_reading=0,
          expected=[1, 2, 1, 2, 1, 2, 2, 2, 2],
      ),
      dict(
          testcase_name="record_producer_fn",
          get_element_producer_fn=RoundrobinRecordElementProducerFn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=3),
          worker_index_to_start_reading=0,
          expected=[
              record.Record(record.RecordMetadata(i), i) for i in range(5)
          ],
      ),
  )
  def test_produces_correct_data(
      self,
      get_element_producer_fn: gp.GetElementProducerFn,
      multiprocessing_options: MultiprocessingOptions,
      worker_index_to_start_reading: int,
      expected: Any,
  ):
    with gp.MultiProcessIterator(
        get_element_producer_fn,
        multiprocessing_options,
        worker_index_to_start_reading,
    ) as iterator:
      actual = list(iterator)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="two_workers",
          get_element_producer_fn=_make_uniform_element_producer_fn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=2),
          worker_index_to_start_reading=1,
          num_iters=5,
          expected_last_worker_index=1,
      ),
      dict(
          testcase_name="five_workers",
          get_element_producer_fn=_make_uniform_element_producer_fn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=5),
          worker_index_to_start_reading=0,
          num_iters=7,
          expected_last_worker_index=1,
      ),
      dict(
          testcase_name="five_workers_incomplete_round",
          get_element_producer_fn=_make_uniform_element_producer_fn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=5),
          worker_index_to_start_reading=0,
          num_iters=3,
          expected_last_worker_index=2,
      ),
      dict(
          testcase_name="from_checkpoint",
          get_element_producer_fn=_make_uniform_element_producer_fn(5),
          multiprocessing_options=MultiprocessingOptions(num_workers=2),
          worker_index_to_start_reading=0,
          num_iters=3,
          expected_last_worker_index=0,
      ),
      dict(
          testcase_name="non_uniform_record_producer_fn",
          get_element_producer_fn=NonUniformElementProducerFn(),
          multiprocessing_options=MultiprocessingOptions(num_workers=3),
          worker_index_to_start_reading=0,
          num_iters=6,
          expected_last_worker_index=2,
      ),
  )
  def test_get_state(
      self,
      get_element_producer_fn: gp.GetElementProducerFn,
      multiprocessing_options: MultiprocessingOptions,
      worker_index_to_start_reading: int,
      num_iters: int,
      expected_last_worker_index: int,
  ):
    with gp.MultiProcessIterator(
        get_element_producer_fn,
        multiprocessing_options,
        worker_index_to_start_reading,
    ) as iterator:
      for _ in range(num_iters):
        _ = next(iterator)
      actual_last_worker_index = iterator.get_last_worker_index()
    self.assertEqual(actual_last_worker_index, expected_last_worker_index)

  def test_fails_with_zero_workers(self):
    with self.assertRaisesRegex(
        ValueError, "Number of processes must be at least 1"
    ):
      with gp.MultiProcessIterator(
          _make_uniform_element_producer_fn(),
          MultiprocessingOptions(num_workers=0),
          0,
      ) as iterator:
        list(iterator)

  def test_propagates_error(self):
    error_msg = "very unique error"

    class FailingGetElementProducerFn(gp.GetElementProducerFn):

      def __call__(
          self, *, worker_index: int, worker_count: int, **kwargs
      ) -> Iterator[int]:
        del self, worker_index, worker_count
        raise ValueError(error_msg)

    failing_get_element_producer_fn = FailingGetElementProducerFn()

    with gp.MultiProcessIterator(
        failing_get_element_producer_fn,
        MultiprocessingOptions(num_workers=2),
        0,
    ) as iterator:
      with self.assertRaisesRegex(gp.RemoteWorkerError, error_msg):
        list(iterator)

  def test_reports_worker_crash(self):

    class FailingGetElementProducerFn(gp.GetElementProducerFn):

      def __call__(
          self, *, worker_index: int, worker_count: int, **kwargs
      ) -> Iterator[int]:
        del self, worker_index, worker_count
        sys.exit(12)

    failing_get_element_producer_fn = FailingGetElementProducerFn()

    with gp.MultiProcessIterator(
        failing_get_element_producer_fn,
        MultiprocessingOptions(num_workers=2),
        0,
    ) as iterator:
      with self.assertRaisesRegex(
          RuntimeError, "was terminated unexpectedly with exit code 12"
      ):
        list(iterator)

  def test_reports_unpicklable_element_producer_fn(self):
    error_msg = "UnpicklableObject is not picklable"

    class UnpicklableObject:

      def __getstate__(self):
        raise ValueError(error_msg)

    local_state = UnpicklableObject()

    class GetElementProducerFnWithUnpicklableClosure(gp.GetElementProducerFn):

      def __call__(
          self, *, worker_index: int, worker_count: int, **kwargs
      ) -> Iterator[int]:
        del self, worker_index, worker_count
        yield 1 if local_state is None else 2

    get_element_producer_fn_with_unpicklable_closure = (
        GetElementProducerFnWithUnpicklableClosure()
    )

    with gp.MultiProcessIterator(
        get_element_producer_fn_with_unpicklable_closure,
        MultiprocessingOptions(num_workers=2),
        0,
    ) as iterator:
      with self.assertRaisesRegex(ValueError, error_msg):
        list(iterator)

  def test_worker_init_fn(self):

    def _set_worker_index_and_count(worker_index: int, worker_count: int):
      gp.monkey_patched_index_and_count = (worker_index, worker_count)

    class GetElementProducerFnReturningGlobal(gp.GetElementProducerFn):

      def __call__(
          self, *, worker_index: int, worker_count: int, **kwargs
      ) -> Iterator[tuple[int, int]]:
        del self, worker_index, worker_count
        yield gp.monkey_patched_index_and_count  # pytype: disable=module-attr

    with gp.MultiProcessIterator(
        GetElementProducerFnReturningGlobal(),
        MultiprocessingOptions(num_workers=2),
        0,
        worker_init_fn=_set_worker_index_and_count,
    ) as iterator:
      result = list(iterator)
    self.assertEqual(result, [(0, 2), (1, 2)])


if __name__ == "__main__":
  absltest.main()
