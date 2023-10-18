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
"""Tests for GrainPool fork."""

from collections.abc import Iterable
import functools
import os
import signal
import time

from absl.testing import absltest
from grain._src.core import sharding
import multiprocessing as mp
from grain._src.python import samplers
import grain._src.python.lazy_dataset.grain_pool_v2 as gp
import jax
from jax import random
import numpy as np


# Functions needs to be defined at the top level in order to be picklable.
class TestSamplerIteratorForProcess:
  """Process-specific iterator for sampler."""

  def __init__(
      self, process_idx: int, num_workers: int, sampler: samplers.Sampler
  ):
    self.record_index = process_idx
    self.num_workers = num_workers
    self.sampler = sampler

  def __iter__(self):
    return self

  def __next__(self) -> int:
    idx = self.record_index
    self.record_index += self.num_workers
    try:
      record_metadata = self.sampler[idx]
      return record_metadata.index
    except Exception as e:
      raise StopIteration from e


def sample_transformation_function(
    process_idx: int, num_workers: int, sampler: samplers.Sampler
) -> TestSamplerIteratorForProcess:
  return TestSamplerIteratorForProcess(process_idx, num_workers, sampler)


def function_taking_some_time(input_iterator: Iterable[int]):
  time.sleep(1)
  return input_iterator


def add_random_value(
    input_iterator: Iterable[int],
    jax_prng_key: jax.Array,
    process_idx: int,
) -> Iterable[float]:  # pylint: disable=unused-argument
  del process_idx
  return map(
      lambda x: x + np.asarray(random.uniform(jax_prng_key)), input_iterator
  )


class GrainPoolTest(absltest.TestCase):

  def test_pool_equal_split(self):
    ctx = mp.get_context("spawn")
    # 16 elements divide equally among 4 processes
    iterable = list(range(16))
    num_processes = 4
    elements_to_buffer = 1
    output_elements = []
    sampler = samplers.IndexSampler(
        num_records=16,
        shard_options=sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
    )
    lazy_ds_worker_function = functools.partial(
        sample_transformation_function,
        num_workers=num_processes,
        sampler=sampler,
    )
    with gp.GrainPool(
        ctx=ctx,
        lazy_ds_worker_function=lazy_ds_worker_function,
        num_processes=num_processes,
        elements_to_buffer_per_process=elements_to_buffer,
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
    expected_elements = list(
        map(lambda x: gp.GrainPoolElement(x, x % num_processes), iterable)
    )
    self.assertEqual(expected_elements, output_elements)
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool.processes, num_processes)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool.processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_non_equal_split(self):
    ctx = mp.get_context("spawn")
    # 14 elements do not divide equally among 4 processes
    iterable = list(range(14))
    num_processes = 4
    elements_to_buffer = 1
    output_elements = []
    sampler = samplers.IndexSampler(
        num_records=14,
        shard_options=sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
    )
    lazy_ds_worker_function = functools.partial(
        sample_transformation_function,
        num_workers=num_processes,
        sampler=sampler,
    )
    with gp.GrainPool(
        ctx=ctx,
        lazy_ds_worker_function=lazy_ds_worker_function,
        num_processes=num_processes,
        elements_to_buffer_per_process=elements_to_buffer,
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
    expected_elements = list(
        map(lambda x: gp.GrainPoolElement(x, x % num_processes), iterable)
    )
    self.assertEqual(expected_elements, output_elements)
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool.processes, num_processes)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool.processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_kill_child(self):
    ctx = mp.get_context("spawn")
    num_processes = 4
    elements_to_buffer = 1
    with gp.GrainPool(
        ctx=ctx,
        lazy_ds_worker_function=function_taking_some_time,
        num_processes=num_processes,
        elements_to_buffer_per_process=elements_to_buffer,
    ) as grain_pool:
      child_pid = grain_pool.processes[0].pid
      os.kill(child_pid, signal.SIGKILL)

    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool.processes, num_processes)
    self.assertEqual(
        grain_pool.processes[0].exitcode, -1 * signal.SIGKILL.value
    )
    for child_process in grain_pool.processes[1:]:
      self.assertEqual(child_process.exitcode, 0)

  def test_pool_object_deletion(self):
    ctx = mp.get_context("spawn")
    num_processes = 4
    elements_to_buffer = 1
    sampler = samplers.IndexSampler(
        num_records=16,
        shard_options=sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
    )
    lazy_ds_worker_function = functools.partial(
        sample_transformation_function,
        num_workers=num_processes,
        sampler=sampler,
    )

    # Users should generally use the with statement, here we test if GrainPool
    # was created without the "with statement", that object deletion would
    # have child processes gracefully exited.
    grain_pool = gp.GrainPool(
        ctx=ctx,
        lazy_ds_worker_function=lazy_ds_worker_function,
        num_processes=num_processes,
        elements_to_buffer_per_process=elements_to_buffer,
    )

    child_processes = grain_pool.processes
    grain_pool.__del__()

    for child_process in child_processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_pickling_jax_objects(self):
    ctx = mp.get_context("spawn")
    iterable = list(range(16))
    num_processes = 4
    elements_to_buffer = 1
    output_elements = []
    lazy_ds_worker_function = functools.partial(
        add_random_value,
        input_iterator=iter(range(int(16 / num_processes))),
        jax_prng_key=jax.random.PRNGKey(0),
    )
    with gp.GrainPool(
        ctx=ctx,
        lazy_ds_worker_function=lazy_ds_worker_function,
        num_processes=num_processes,
        elements_to_buffer_per_process=elements_to_buffer,
    ) as grain_pool:
      for element in grain_pool:
        output_elements.append(element)
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool.processes, num_processes)
    self.assertLen(output_elements, len(iterable))
    # Make sure all child processes exited successfully.
    for child_process in grain_pool.processes:
      self.assertEqual(child_process.exitcode, 0)


if __name__ == "__main__":
  absltest.main()
