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
"""Tests for data sources."""

from collections.abc import Sequence
import dataclasses
import pathlib
import pickle
import random
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import multiprocessing as grain_multiprocessing
from grain._src.python import data_sources

FLAGS = flags.FLAGS


@dataclasses.dataclass
class DummyFileInstruction:
  filename: str
  skip: int
  take: int
  examples_in_shard: int


class DataSourceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir)


class RangeDataSourceTest(DataSourceTest):

  @parameterized.parameters([
      (0, 10, 2),  # Positive step
      (10, 0, -2),  # Negative step
      (0, 0, 1),  # Empty Range
  ])
  def test_range_data_source(self, start, stop, step):
    expected_output = list(range(start, stop, step))

    range_ds = data_sources.RangeDataSource(start, stop, step)
    actual_output = [range_ds[i] for i in range(len(range_ds))]

    self.assertEqual(expected_output, actual_output)


class InMemoryDataSourceTest(DataSourceTest):

  def test_single_process(self):
    sequence = list(range(12))
    in_memory_ds = data_sources.SharedMemoryDataSource(sequence)

    output_by_index = [in_memory_ds[i] for i in range(len(in_memory_ds))]
    self.assertEqual(sequence, output_by_index)

    output_by_list = list(in_memory_ds)
    self.assertEqual(sequence, output_by_list)

    in_memory_ds.close()
    in_memory_ds.unlink()

  @staticmethod
  def read_elements(
      in_memory_ds: data_sources.SharedMemoryDataSource, indices: Sequence[int]
  ) -> Sequence[Any]:
    res = [in_memory_ds[i] for i in indices]
    return res

  def test_multi_processes_co_read(self):
    sequence = list(range(12))
    in_memory_ds = data_sources.SharedMemoryDataSource(
        sequence, name="DataSourceTestingCoRead"
    )

    num_processes = 3
    indices_for_processes = [[1, 3, 5], [2, 3, 4], [5, 2, 3]]
    expected_elements_read = list(
        map(
            lambda indices: [in_memory_ds[i] for i in indices],
            indices_for_processes,
        )
    )

    mp_context = grain_multiprocessing.get_context("spawn")
    with mp_context.Pool(processes=num_processes) as pool:
      elements_read = pool.starmap(
          InMemoryDataSourceTest.read_elements,
          zip([in_memory_ds] * num_processes, indices_for_processes),
      )

      pool.close()
      pool.join()

    self.assertEqual(elements_read, expected_elements_read)

    in_memory_ds.close()
    in_memory_ds.unlink()

  def test_empty_sequence(self):
    in_memory_ds = data_sources.SharedMemoryDataSource([])
    self.assertEmpty(in_memory_ds)

    in_memory_ds.close()
    in_memory_ds.unlink()

  def test_str(self):
    sequence = list(range(12))
    name = "DataSourceTestingStr"
    in_memory_ds = data_sources.SharedMemoryDataSource(sequence, name=name)
    actual_str = str(in_memory_ds)
    self.assertEqual(
        actual_str,
        f"InMemoryDataSource(name={name}, len={len(sequence)})",
    )

    in_memory_ds.close()
    in_memory_ds.unlink()


class ArrayRecordDataSourceTest(DataSourceTest):

  def test_array_record_data_implements_random_access(self):
    assert issubclass(
        data_sources.ArrayRecordDataSource, data_sources.RandomAccessDataSource
    )

  def test_array_record_source_empty_sequence(self):
    with self.assertRaises(ValueError):
      data_sources.ArrayRecordDataSource([])


if __name__ == "__main__":
  absltest.main()
