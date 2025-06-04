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
"""Tests for data loader."""

from collections.abc import Sequence
import pathlib
import sys
from typing import Union
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized as absl_parameterized
from grain._src.core import sharding
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import data_loader as data_loader_lib
from grain._src.python import options
from grain._src.python import samplers
from grain._src.python import shared_memory_array
# pylint: disable=g-importing-member
from grain._src.python.data_sources import ArrayRecordDataSource
from grain._src.python.data_sources import RangeDataSource
from grain._src.python.data_sources import SharedMemoryDataSource
from grain._src.python.operations import BatchOperation
from grain._src.python.operations import FilterOperation
from grain._src.python.operations import MapOperation
# pylint: enable=g-importing-member
import numpy as np
import parameterized


FLAGS = flags.FLAGS


def map_function(data):
  return data + 1


def condition_function(data):
  return data % 2 == 0


class FilterEven(transforms.Filter):

  def filter(self, x: int) -> bool:
    return x % 2 == 0


class PlusOne(transforms.MapTransform):

  def map(self, x: int) -> int:
    return x + 1


class PlusRandom(transforms.RandomMapTransform):

  def random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + rng.integers(100_000)


class FailingMap(transforms.MapTransform):

  def map(self, x):
    del x
    1 / 0  # pylint: disable=pointless-statement


class NonPickableTransform(transforms.MapTransform):

  def __getstate__(self):
    raise ValueError("I shall not be pickled")

  def map(self, x):
    return x


class RaisingTransform(transforms.MapTransform):

  def map(self, x):
    raise AttributeError("I shall raise")


class ExitingTransform(transforms.MapTransform):

  def map(self, x):
    raise sys.exit(123)


class RandomTripletSource:

  def __len__(self) -> int:
    return 100_000

  def __getitem__(self, record_key: int):
    return {
        "data": np.random.uniform(size=(3, 224, 224, 3)).astype(
            dtype=np.float32
        )
    }


class CopyNumPyArrayToSharedMemoryTest(absltest.TestCase):

  def test_copy_numpy_array_to_shared_memory(self):
    element = np.array([1, 2, 3, 4, 5, 6, 7])
    transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
    result = transform.map(element)
    self.assertIsInstance(result, shared_memory_array.SharedMemoryArrayMetadata)

  def test_copy_nested_numpy_array_to_shared_memory(self):
    element_1 = np.arange(5)
    element_2 = np.arange(5)
    transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
    result = transform.map([element_1, element_2])
    self.assertIsInstance(
        result[0], shared_memory_array.SharedMemoryArrayMetadata
    )
    self.assertIsInstance(
        result[1], shared_memory_array.SharedMemoryArrayMetadata
    )

  def test_copy_skipped_non_numpy_array(self):
    element = "randomstring"
    transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
    result = transform.map(element)
    self.assertIs(result, element)

  def test_copy_skipped_dtype_hasobject(self):
    class DT:
      pass

    element = np.array([127, 128, 129], dtype=np.dtype(DT))
    transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
    result = transform.map(element)
    print(result)
    self.assertIs(result, element)

  def test_copy_skipped_flags_c_contiguous(self):
    element = np.arange(9).reshape(3, 3)[:, (0, 1)]
    transform = data_loader_lib.CopyNumPyArrayToSharedMemory()
    result = transform.map(element)
    self.assertIs(result, element)


@parameterized.parameterized_class([
    {"num_threads_per_worker": None},
    {"num_threads_per_worker": 0},
    {"num_threads_per_worker": 15},
])
class DataLoaderTest(absl_parameterized.TestCase):
  # Number of prefetch threads for each Grain worker
  num_threads_per_worker: int | None

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir) / "testdata"
    self.read_options = (
        options.ReadOptions(num_threads=self.num_threads_per_worker)
        if (self.num_threads_per_worker is not None)
        else None
    )

  def _create_data_loader_for_short_sequence(
      self,
      transformations,
      *,
      worker_count: int = 0,
      seed: Union[int, None] = None,
  ) -> data_loader_lib.DataLoader:
    # Generates elements [0, 1, 2, 3, 4, 5, 6, 7].
    range_data_source = RangeDataSource(start=0, stop=8, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.NoSharding(),
        seed=seed,
    )
    return data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=transformations,
        worker_count=worker_count,
        read_options=self.read_options,
    )

  def test_fails_to_pickle(self):
    transformations = [NonPickableTransform()]
    data_loader = self._create_data_loader_for_short_sequence(
        transformations, worker_count=2
    )
    with self.assertRaisesRegex(ValueError, "I shall not be pickled"):
      list(data_loader)

  def test_propagates_transform_error_with_multiprocessing(self):
    transformations = [RaisingTransform()]
    data_loader = self._create_data_loader_for_short_sequence(
        transformations, worker_count=2
    )
    with self.assertRaisesRegex(Exception, "I shall raise"):
      list(data_loader)

  def test_reports_multiprocessing_worker_crash(self):
    transformations = [ExitingTransform()]
    data_loader = self._create_data_loader_for_short_sequence(
        transformations, worker_count=2
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "was terminated unexpectedly with exit code 123",
    ):
      list(data_loader)

  def test_data_loader_single_process(self):
    # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
    # Filter keeps only even elements [2, 4, 6, 8]
    # Batching batches each 2 consective elements, producing
    # [np.array([2, 4]), np.array([6, 8])]
    transformations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]
    data_loader = self._create_data_loader_for_short_sequence(transformations)
    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  def test_data_loader_single_process_random_map(self):
    transformations = [
        PlusRandom(),
        BatchOperation(batch_size=2),
    ]
    data_loader = self._create_data_loader_for_short_sequence(
        transformations, seed=1
    )
    actual = list(data_loader)
    # 4 batches of size 2.
    self.assertLen(actual, 4)
    for i in range(4):
      self.assertEqual(actual[i].shape, (2,))

  def test_data_loader_single_process_legacy_operations(self):
    """Test that old style operations that implement __call__() still work."""
    transformations = [
        MapOperation(map_function=map_function),
        FilterOperation(condition_function=condition_function),
        BatchOperation(batch_size=2),
    ]
    data_loader = self._create_data_loader_for_short_sequence(transformations)
    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  def test_data_loader_single_process_iterate_twice(self):
    transformations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]
    data_loader = self._create_data_loader_for_short_sequence(transformations)
    expected = [np.array([2, 4]), np.array([6, 8])]
    # First iteration.
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)
    # Second iteration.
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  def test_data_loader_in_memory_data_source(self):
    data_source = SharedMemoryDataSource([0, 1, 2, 3, 4, 5, 6, 7])

    sampler = samplers.SequentialSampler(
        num_records=len(data_source), shard_options=sharding.NoSharding()
    )

    # Multiprocessing (with 2 processes), splits elements such that:
    # Process_0 gets [0, 2, 4, 6]
    # Process_1 gets [1, 3, 5, 7]
    # Afterwards, operations are executed on elements from each process.
    operations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]

    num_workers = 2
    data_loader = data_loader_lib.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        read_options=self.read_options,
    )

    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)

    np.testing.assert_equal(actual, expected)

  def test_data_loader_two_processes_no_shared_memory(self):
    # Generates elements [0, 1, 2, 3, 4, 5, 6, 7]
    range_data_source = RangeDataSource(start=0, stop=8, step=1)

    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )

    # Multiprocessing (with 2 processes), splits elements such that:
    # Process_0 gets [0, 2, 4, 6]
    # Process_1 gets [1, 3, 5, 7]
    # Afterwards, operations are executed on elements from each process.
    operations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]

    num_workers = 2
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        read_options=self.read_options,
    )

    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)

    np.testing.assert_equal(actual, expected)

  def test_data_loader_two_processes_with_shared_memory(self):
    # Generates elements [0, 1, 2, 3, 4, 5, 6, 7]
    range_data_source = RangeDataSource(start=0, stop=8, step=1)

    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )

    # Multiprocessing (with 2 processes), splits elements such that:
    # Process_0 gets [0, 2, 4, 6]
    # Process_1 gets [1, 3, 5, 7]
    # Afterwards, operations are executed on elements from each process.
    operations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]

    num_workers = 2
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        read_options=self.read_options,
    )

    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)

    np.testing.assert_equal(actual, expected)

  def test_data_loader_remote_exception(self):
    range_data_source = RangeDataSource(start=0, stop=8, step=1)

    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )

    operations = [FailingMap()]

    num_workers = 2
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        read_options=self.read_options,
    )
    with self.assertRaises(Exception) as e:
      list(data_loader)
      assert "ZeroDivisionError: division by zero" in e.__cause__._traceback  # pytype: disable=attribute-error

  def test_data_loader_with_used_array_record_data_source(
      self,
  ):
    data_source = ArrayRecordDataSource([
        str(self.testdata_dir / "digits.array_record-00000-of-00002"),
        str(self.testdata_dir / "digits.array_record-00001-of-00002"),
    ])

    data_source[0]  # pylint: disable=pointless-statement

    sampler = samplers.SequentialSampler(
        num_records=len(data_source), shard_options=sharding.NoSharding()
    )

    num_workers = 1
    data_loader = data_loader_lib.DataLoader(
        data_source=data_source,
        sampler=sampler,
        worker_count=num_workers,
        read_options=self.read_options,
    )
    expected = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
    actual = list(data_loader)

    np.testing.assert_equal(actual, expected)

  def test_data_loader_with_invalid_number_of_workers(self):
    """Test a value error is raised when an invlaid number of workers is used."""
    ar_data_source = ArrayRecordDataSource([
        str(self.testdata_dir / "digits.array_record-00000-of-00002"),
        str(self.testdata_dir / "digits.array_record-00001-of-00002"),
    ])

    sampler = samplers.SequentialSampler(
        num_records=len(ar_data_source), shard_options=sharding.NoSharding()
    )

    num_workers = -1
    with self.assertRaises(ValueError):
      data_loader_lib.DataLoader(
          data_source=ar_data_source,
          sampler=sampler,
          worker_count=num_workers,
          read_options=self.read_options,
      )

  def create_checkpointing_dataloader(
      self, num_workers: int
  ) -> data_loader_lib.DataLoader:
    """Creates a DataLoader object for checkpointing tests."""
    range_data_source = RangeDataSource(start=0, stop=16, step=1)
    sampler = samplers.IndexSampler(
        num_records=len(range_data_source),
        shard_options=sharding.NoSharding(),
        shuffle=False,
        num_epochs=1,
    )
    operations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(batch_size=2),
    ]
    return data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        read_options=self.read_options,
    )

  @absl_parameterized.parameters(
      {
          "num_workers": 0,
          "steps_to_iterate": 0,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 0,
          "steps_to_iterate": 1,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 2,
          "steps_to_iterate": 1,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 2,
          "steps_to_iterate": 2,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 3,
          "steps_to_iterate": 2,
          "expected": [
              np.array([4, 10]),
              np.array([2, 8]),
              np.array([6, 12]),
              np.array(16),
              np.array([14]),
          ],
      },
      {
          "num_workers": 3,
          "steps_to_iterate": 3,
          "expected": [
              np.array([4, 10]),
              np.array([2, 8]),
              np.array([6, 12]),
              np.array(16),
              np.array([14]),
          ],
      },
  )
  def test_data_loader_checkpointing_object_reconstruction(
      self,
      num_workers: int,
      steps_to_iterate: int,
      expected: Sequence[np.ndarray],
  ):
    data_loader_iterator = iter(
        self.create_checkpointing_dataloader(num_workers)
    )

    # actual contains elements obtained by iterating through dataloader before
    # getting state, as well as after state is restored. Should be identical
    # to elements obtained by iterating without checkpointing (expected.)
    actual = [next(data_loader_iterator) for i in range(steps_to_iterate)]

    state = data_loader_iterator.get_state()

    # Advance the iterator after getting the state. After restoring the iterator
    # to the state above, the element should appear again when iterating.
    np.testing.assert_equal(
        next(data_loader_iterator), expected[steps_to_iterate]
    )

    # Create new objects (similar to after preemption) and attempt to restore
    # checkpointed state into them.
    restored_data_loader = self.create_checkpointing_dataloader(num_workers)
    restored_data_loader_iterator = iter(restored_data_loader)
    restored_data_loader_iterator.set_state(state)

    for item in restored_data_loader_iterator:
      actual.append(item)

    np.testing.assert_equal(actual, expected)

  @absl_parameterized.parameters(
      {
          "num_workers": 0,
          "steps_to_iterate": 0,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 2,
          "steps_to_iterate": 0,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 2,
          "steps_to_iterate": 1,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 2,
          "steps_to_iterate": 2,
          "expected": [
              np.array([2, 4]),
              np.array([6, 8]),
              np.array([10, 12]),
              np.array([14, 16]),
          ],
      },
      {
          "num_workers": 3,
          "steps_to_iterate": 0,
          "expected": [
              np.array([4, 10]),
              np.array([2, 8]),
              np.array([6, 12]),
              np.array(16),
              np.array([14]),
          ],
      },
      {
          "num_workers": 3,
          "steps_to_iterate": 2,
          "expected": [
              np.array([4, 10]),
              np.array([2, 8]),
              np.array([6, 12]),
              np.array(16),
              np.array([14]),
          ],
      },
      {
          "num_workers": 3,
          "steps_to_iterate": 3,
          "expected": [
              np.array([4, 10]),
              np.array([2, 8]),
              np.array([6, 12]),
              np.array(16),
              np.array([14]),
          ],
      },
  )
  def test_data_loader_checkpointing_same_object(
      self,
      num_workers: int,
      steps_to_iterate: int,
      expected: Sequence[np.ndarray],
  ):
    data_loader_iterator = iter(
        self.create_checkpointing_dataloader(num_workers)
    )

    # actual contains elements obtained by iterating through dataloader before
    # getting state, as well as after state is restored. Should be identical
    # to elements obtained by iterating without checkpointing (expected.)
    actual = [next(data_loader_iterator) for i in range(steps_to_iterate)]

    state = data_loader_iterator.get_state()

    # Advance the iterator after getting the state. After restoring the iterator
    # to the state above, the element should appear again when iterating.
    np.testing.assert_equal(
        next(data_loader_iterator), expected[steps_to_iterate]
    )

    data_loader_iterator.set_state(state)
    for item in data_loader_iterator:
      actual.append(item)
    np.testing.assert_equal(actual, expected)

  def test_batch_transform_mapped_to_batch_operation(self):
    # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
    # Filter keeps only even elements [2, 4, 6, 8]
    # Batching batches each 2 consective elements, producing
    # [np.array([2, 4]), np.array([6, 8])]
    transformations = [
        PlusOne(),
        FilterEven(),
        transforms.Batch(batch_size=2),
    ]
    data_loader = self._create_data_loader_for_short_sequence(transformations)
    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  @mock.patch.object(data_loader_lib, "CopyNumPyArrayToSharedMemory")
  def test_shared_memory(self, mock_copy_numpy_array_to_shared_memory):
    range_data_source = RangeDataSource(start=0, stop=8, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )

    operations = [
        PlusOne(),
        FilterEven(),
    ]

    batch_operation = mock.MagicMock(BatchOperation(batch_size=2))

    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        worker_count=0,
        read_options=self.read_options,
    )
    batch_operation._enable_shared_memory.assert_not_called()
    self.assertTrue(
        data_loader._operations[-1], mock_copy_numpy_array_to_shared_memory
    )

    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations + [batch_operation],
        worker_count=2,
        read_options=self.read_options,
    )
    batch_operation._enable_shared_memory.assert_called_once()
    self.assertTrue(data_loader._operations[-1], batch_operation)

  def test_state_without_in_memory_data(self):
    data_source = list(range(10000))
    loader = data_loader_lib.DataLoader(
        data_source=data_source,
        sampler=samplers.SequentialSampler(num_records=len(data_source)),
        read_options=self.read_options,
    )
    it = loader.__iter__()
    state = it.get_state()
    self.assertLess(len(state), 1000)


class PyGrainDatasetIteratorTest(absltest.TestCase):

  def test_str(self):
    range_data_source = RangeDataSource(start=0, stop=8, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.NoSharding(),
        seed=1,
    )
    loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        worker_count=3,
    )
    itr = iter(loader)

    expected_str = """PyGrainDatasetIterator(state={
    "version": 2,
    "last_seen_indices": {
        "0": -3,
        "1": -2,
        "2": -1
    },
    "last_worker_index": -1,
    "worker_count": 3,
    "sampler": "SequentialSampler(num_records=8, shard_options=NoSharding(shard_index=0, shard_count=1, drop_remainder=False))",
    "data_source": "RangeDataSource(start=0, stop=8, step=1)"
})"""

    self.assertEqual(expected_str, str(itr))


if __name__ == "__main__":
  absltest.main()
