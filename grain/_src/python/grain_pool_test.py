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

import os
import signal

from absl.testing import absltest
import multiprocessing as mp
from grain._src.python import data_sources
from grain._src.python import grain_pool as gp
from grain._src.python.options import MultiprocessingOptions  # pylint: disable=g-importing-member


class GrainPoolTest(absltest.TestCase):

  def test_pool_equal_split_in_memory_data_source(self):
    in_memory_ds = data_sources.InMemoryDataSource(range(12))

    # 12 elements in the `in_memory_ds` are divided
    # equally among 4 processes.
    def get_element_producer_fn(worker_index: int, worker_count: int):
      return iter(range(worker_index, 12, worker_count))

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
    def get_element_producer_fn(worker_index: int, worker_count: int):
      return iter(range(worker_index, 16, worker_count))

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
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_non_equal_split(self):
    ctx = mp.get_context("spawn")

    # 14 elements do not divide equally among 4 processes
    def get_element_producer_fn(worker_index: int, worker_count: int):
      return iter(range(worker_index, 14, worker_count))

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
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_kill_child(self):
    ctx = mp.get_context("spawn")

    def get_element_producer_fn(worker_index: int, worker_count: int):
      return iter(range(worker_index, 14, worker_count))

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
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_object_deletion(self):
    ctx = mp.get_context("spawn")

    def get_element_producer_fn(worker_index: int, worker_count: int):
      return iter(range(worker_index, 14, worker_count))

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
      self.assertEqual(child_process.exitcode, 0)


if __name__ == "__main__":
  absltest.main()
