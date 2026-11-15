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
import functools
import inspect
import json
from multiprocessing import shared_memory
import pathlib
import pickle
import platform
import sys
import threading
from typing import Any, Union
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
from grain._src.python.data_sources import ArrayRecordDataSource
from grain._src.python.data_sources import RangeDataSource
from grain._src.python.data_sources import SharedMemoryDataSource
from grain._src.python.dataset.transformations import batch
from grain._src.python.dataset.transformations import process_prefetch
from grain._src.python.dataset.transformations import source
from grain._src.python.ipc import shared_memory_array
from grain._src.python.operations import BatchOperation
from grain._src.python.operations import FilterOperation
from grain._src.python.operations import MapOperation
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np
import parameterized


FLAGS = flags.FLAGS


def setup_module():
  # Set the path to test data when run via pytest.
  # When run via bazel, FLAGS.test_srcdir is set from the
  # BUILD file, see args = ["--test_srcdir=grain/_src/python"]
  # in grain/_src/python/BUILD
  import grain  # pylint: disable=g-import-not-at-top

  srcdir = pathlib.Path(grain.__file__).parents[0] / "_src" / "python"
  FLAGS["test_srcdir"].parse(str(srcdir))


def map_function(data):
  return data + 1


def condition_function(data):
  return data % 2 == 0


class FilterEven(transforms.Filter):

  def filter(self, x: int) -> bool:
    return x % 2 == 0


class PlusOne(transforms.Map):

  def map(self, x: int) -> int:
    return x + 1


class PlusRandom(transforms.RandomMap):

  def random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + rng.integers(100_000)  # pyrefly: ignore[bad-return]


class FailingMap(transforms.Map):

  def map(self, x):
    del x
    1 / 0  # pylint: disable=pointless-statement


class NonPickableTransform(transforms.Map):

  def __getstate__(self):
    raise ValueError("I shall not be pickled")

  def map(self, x):
    return x


class RaisingTransform(transforms.Map):

  def map(self, x):
    raise AttributeError("I shall raise")


class ExitingTransform(transforms.Map):

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


class DuplicateElementFlatMap(transforms.FlatMap):
  max_fan_out: int = 7

  def flat_map(self, element: Any) -> Any:
    for _ in range(self.max_fan_out):
      yield element


class PlusTfRandom(transforms.TfRandomMap):

  def np_random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + int(rng.integers(10))


class BrokenReprArrayMap(transforms.Map):

  def __init__(self):
    self.weights = np.array([1, 2, 3])

  def map(self, x: int) -> int:
    return x + int(self.weights[0])

  def __repr__(self) -> str:
    raise RuntimeError("broken map repr")

  def __str__(self) -> str:
    raise RuntimeError("broken map str")


class BrokenReprArrayFilter(transforms.Filter):

  def __init__(self):
    self.mask = np.array([True, False])

  def filter(self, x: int) -> bool:
    return bool(self.mask[0]) and x >= 0

  def __repr__(self) -> str:
    raise RuntimeError("broken filter repr")

  def __str__(self) -> str:
    raise RuntimeError("broken filter str")


class BrokenReprArrayRandomMap(transforms.RandomMap):

  def __init__(self):
    self.offsets = np.array([0, 1])

  def random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + int(self.offsets[int(rng.integers(2))])

  def __repr__(self) -> str:
    raise RuntimeError("broken random_map repr")

  def __str__(self) -> str:
    raise RuntimeError("broken random_map str")


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


@absltest.skipIf(
    platform.system() == "Windows", "Skipped due to Windows paths."
)
@parameterized.parameterized_class([
    {"num_threads_per_worker": None},
    {"num_threads_per_worker": 0},
    {"num_threads_per_worker": 15},
])
class DataLoaderTest(absl_parameterized.TestCase):

  def tearDown(self):
    super().tearDown()

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
        data_source=data_source,  # pyrefly: ignore[bad-argument-type]
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
      assert "ZeroDivisionError: division by zero" in e.__cause__._traceback  # pyrefly: ignore[missing-attribute]

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

  @absl_parameterized.parameters(
      {"num_workers": 0}, {"num_workers": 1}, {"num_workers": 2}
  )
  def test_data_loader_sequential_checkpoint_restore_drift(
      self, num_workers: int
  ):
    data_loader_iterator = iter(
        self.create_checkpointing_dataloader(num_workers=num_workers)
    )

    # Fetch the first element to have nontrivial initial state.
    first_element = next(data_loader_iterator)
    np.testing.assert_equal(first_element, np.array([2, 4]))

    # Perform sequential get/set state operations without advancing.
    state1 = data_loader_iterator.get_state()
    data_loader_iterator.set_state(state1)

    state2 = data_loader_iterator.get_state()
    data_loader_iterator.set_state(state2)

    # Assert that we can still advance the iterator and that the state is
    # still the same after all these operations.
    next_element = next(data_loader_iterator)
    np.testing.assert_equal(next_element, np.array([6, 8]))

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

  def test_data_loader_with_batch_fn(self):
    # Map transforms elements to be [1, 2, 3, 4, 5, 6, 7, 8]
    # Filter keeps only even elements [2, 4, 6, 8]
    # Batching batches each 3 consective elements with batch_and_pad fn,
    # producing [np.array([2, 4, 6]), np.array([8, 0, 0])]
    transformations = [
        PlusOne(),
        FilterEven(),
        BatchOperation(
            batch_size=3,
            batch_fn=functools.partial(batch.batch_and_pad, batch_size=3),
        ),
    ]
    data_loader = self._create_data_loader_for_short_sequence(transformations)
    expected = [np.array([2, 4, 6]), np.array([8, 0, 0])]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  def test_state_without_in_memory_data(self):
    data_source = list(range(10000))
    loader = data_loader_lib.DataLoader(
        data_source=data_source,  # pyrefly: ignore[bad-argument-type]
        sampler=samplers.SequentialSampler(num_records=len(data_source)),
        read_options=self.read_options,
    )
    it = loader.__iter__()
    state = it.get_state()
    self.assertLess(len(state), 1000)

  def test_data_loader_with_flat_map(self):
    range_data_source = RangeDataSource(start=0, stop=8, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )
    operations = [
        PlusOne(),
        FilterEven(),
        DuplicateElementFlatMap(),
    ]
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        read_options=self.read_options,
    )
    np.testing.assert_equal(
        list(data_loader), [2] * 7 + [4] * 7 + [6] * 7 + [8] * 7
    )

  def test_data_loader_with_flat_map_checkpointing(self):
    range_data_source = RangeDataSource(start=0, stop=8, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )
    operations = [
        PlusOne(),
        FilterEven(),
        DuplicateElementFlatMap(),
    ]
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        operations=operations,
        read_options=self.read_options,
    )
    assert_equal_output_after_checkpoint(data_loader)

  @absl_parameterized.product(
      worker_count=[0, 4], num_start_prefetch_calls=[1, 5]
  )
  def test_start_prefetch(
      self, worker_count: int, num_start_prefetch_calls: int
  ):
    range_data_source = RangeDataSource(start=0, stop=16, step=1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source), shard_options=sharding.NoSharding()
    )
    data_loader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
        read_options=self.read_options,
        worker_count=worker_count,
    )
    data_loader_iterator = data_loader.__iter__()
    for _ in range(num_start_prefetch_calls):
      data_loader_iterator.start_prefetch()
    self.assertEqual(list(data_loader_iterator), list(range(16)))

  def test_data_loader_sharding_drop_remainder(self):
    range_data_source = RangeDataSource(start=0, stop=10, step=1)

    # Shard 0, drop_remainder=False
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.ShardOptions(
            shard_index=0, shard_count=3, drop_remainder=False
        ),
    )
    dataloader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
    )
    actual = list(dataloader)
    self.assertEqual(actual, [0, 3, 6, 9])
    self.assertLen(actual, 4)

    # Shard 0, drop_remainder=True
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.ShardOptions(
            shard_index=0, shard_count=3, drop_remainder=True
        ),
    )
    dataloader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
    )
    actual = list(dataloader)
    self.assertEqual(actual, [0, 3, 6])
    self.assertLen(actual, 3)

    # Shard 1, drop_remainder=False
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.ShardOptions(
            shard_index=1, shard_count=3, drop_remainder=False
        ),
    )
    dataloader = data_loader_lib.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
    )
    actual = list(dataloader)
    self.assertEqual(actual, [1, 4, 7])
    self.assertLen(actual, 3)

  def test_dataloader_transforms_do_not_call_inspect_getsourcelines(self):
    transformations = [
        PlusOne(),
        FilterEven(),
        PlusRandom(),
        PlusTfRandom(),
        transforms.Batch(2),
    ]
    with mock.patch.object(
        inspect,
        "getsourcelines",
        side_effect=AssertionError("getsourcelines should not be called"),
    ):
      dl_single = self._create_data_loader_for_short_sequence(
          transformations, worker_count=0, seed=42
      )
      it_single = iter(dl_single)
      _ = it_single._iterator._stats
      out_single = list(it_single)
      self.assertNotEmpty(out_single)

      serialized = process_prefetch._serialize_dataset(dl_single._dataset)
      self.assertNotEmpty(serialized)
      for adapter in (
          data_loader_lib._ExtractDataMapAdapter(),
          data_loader_lib._MapTransformAdapter(PlusOne()),
          data_loader_lib._RandomMapTransformAdapter(PlusRandom()),
          data_loader_lib._TfRandomMapTransformAdapter(PlusTfRandom()),
          data_loader_lib._FilterTransformAdapter(FilterEven()),
          data_loader_lib._RecordBatchFn(batch.make_batch),
      ):
        _ = pickle.loads(pickle.dumps(adapter))  # pylint: disable=g-unsafe-pickle-load

      dl_mp = self._create_data_loader_for_short_sequence(
          transformations, worker_count=2, seed=42
      )
      out_mp = list(dl_mp)
      np.testing.assert_equal(out_mp, out_single)

  def test_adapter_does_not_call_broken_transform_repr(self):
    transformations = [
        BrokenReprArrayMap(),
        BrokenReprArrayFilter(),
        BrokenReprArrayRandomMap(),
    ]
    dl = self._create_data_loader_for_short_sequence(
        transformations, worker_count=0, seed=7
    )
    self.assertNotEmpty(str(dl._dataset))
    it = iter(dl)
    _ = it._iterator._stats
    self.assertLen(list(it), 8)


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


class _MutableReprSource(RangeDataSource):

  def __init__(self, start: int, stop: int, step: int = 1, tag: str = "v1"):
    super().__init__(start=start, stop=stop, step=step)
    self.tag = tag
    self.repr_calls = 0

  def __repr__(self) -> str:
    self.repr_calls += 1
    return f"_MutableReprSource(stop={self._stop}, tag={self.tag})"


class _MutableReprSampler(samplers.SequentialSampler):

  def __init__(
      self,
      num_records: int,
      shard_options: sharding.ShardOptions = sharding.NoSharding(),
      tag: str = "v1",
  ):
    super().__init__(num_records=num_records, shard_options=shard_options)
    self.tag = tag
    self.repr_calls = 0

  def __repr__(self) -> str:
    self.repr_calls += 1
    return f"_MutableReprSampler(n={self._num_records}, tag={self.tag})"


class DataLoaderStateFastPathTest(absl_parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.read_options = options.ReadOptions(
        num_threads=0, prefetch_buffer_size=0
    )
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir) / "testdata"

  def test_mutable_custom_source_and_sampler_never_serve_stale_repr(self):
    src_v1 = _MutableReprSource(start=0, stop=8, tag="v1")
    smp_v1 = _MutableReprSampler(num_records=8, tag="v1")
    loader = data_loader_lib.DataLoader(
        data_source=src_v1,
        sampler=smp_v1,
        worker_count=0,
        read_options=self.read_options,
    )
    itr = iter(loader)
    _ = next(itr)
    ckpt_v1 = itr.get_state()
    self.assertIn("tag=v1", ckpt_v1.decode())

    # Mutate custom source in-place: old v1 ckpt must fail, new ckpt emits v2.
    src_v1.tag = "v2"
    with self.assertRaisesRegex(
        ValueError, "DataSource in checkpoint does not match datasource"
    ):
      itr.set_state(ckpt_v1)
    ckpt_src_v2 = itr.get_state()
    self.assertIn("_MutableReprSource(stop=8, tag=v2)", ckpt_src_v2.decode())

    # Mutate custom sampler in-place: old ckpt must fail, new ckpt emits v2.
    smp_v1.tag = "v2"
    with self.assertRaisesRegex(
        ValueError, "Sampler in checkpoint does not match dataloader sampler"
    ):
      itr.set_state(ckpt_src_v2)
    ckpt_both_v2 = itr.get_state()
    self.assertIn("_MutableReprSampler(n=8, tag=v2)", ckpt_both_v2.decode())

    # Fresh v2 loader accepts v2 checkpoint and continues identically.
    loader_v2 = data_loader_lib.DataLoader(
        data_source=_MutableReprSource(start=0, stop=8, tag="v2"),
        sampler=_MutableReprSampler(num_records=8, tag="v2"),
        worker_count=0,
        read_options=self.read_options,
    )
    itr_v2 = iter(loader_v2)
    itr_v2.set_state(ckpt_both_v2)
    self.assertEqual(list(itr_v2), [1, 2, 3, 4, 5, 6, 7])

  def test_builtin_array_record_and_index_sampler_cached_and_invalidated(self):
    paths = [
        str(self.testdata_dir / "digits.array_record-00000-of-00002"),
        str(self.testdata_dir / "digits.array_record-00001-of-00002"),
    ]
    ar_source = ArrayRecordDataSource(paths)
    sampler = samplers.IndexSampler(
        num_records=len(ar_source),
        shard_options=sharding.NoSharding(),
        shuffle=True,
        num_epochs=1,
        seed=42,
    )
    with (
        mock.patch.object(
            ArrayRecordDataSource,
            "__repr__",
            autospec=True,
            wraps=ArrayRecordDataSource.__repr__,
        ) as mock_ar_repr,
        mock.patch.object(
            samplers.IndexSampler,
            "__repr__",
            autospec=True,
            wraps=samplers.IndexSampler.__repr__,
        ) as mock_smp_repr,
    ):
      loader = data_loader_lib.DataLoader(
          data_source=ar_source,
          sampler=sampler,
          worker_count=0,
          read_options=self.read_options,
      )
      itr = iter(loader)
      _ = next(itr)
      ckpt = b""
      for _ in range(10):
        ckpt = itr.get_state()
      for _ in range(5):
        itr.set_state(ckpt)
      self.assertEqual(mock_ar_repr.call_count, 1)
      self.assertEqual(mock_smp_repr.call_count, 1)

      # In-place _paths mutation on ArrayRecordDataSource invalidates cache.
      ar_source._paths[0] = paths[1]
      with self.assertRaisesRegex(
          ValueError, "DataSource in checkpoint does not match datasource"
      ):
        itr.set_state(ckpt)
      self.assertEqual(mock_ar_repr.call_count, 2)

  def test_set_state_encoding_and_malformed_json_rejection_parity(self):
    loader = data_loader_lib.DataLoader(
        data_source=RangeDataSource(start=0, stop=8, step=1),
        sampler=samplers.SequentialSampler(
            num_records=8, shard_options=sharding.NoSharding()
        ),
        worker_count=0,
        read_options=self.read_options,
    )
    itr = iter(loader)
    with self.assertRaises(json.JSONDecodeError):
      itr.set_state(b"\xef\xbb\xbf{}")
    with self.assertRaises(UnicodeDecodeError):
      itr.set_state("{}".encode("utf-16"))
    with self.assertRaises(UnicodeDecodeError):
      itr.set_state(b"\xff")
    with self.assertRaises(json.JSONDecodeError):
      itr.set_state(b"{bad")

  @absl_parameterized.product(
      worker_count=[0, 1, 2],
      shard=[(0, 1), (2, 3)],
      steps=[0, 2],
  )
  def test_get_state_exact_json_parity_across_workers_and_shards(
      self,
      worker_count: int,
      shard: tuple[int, int],
      steps: int,
  ):
    shard_index, shard_count = shard
    shard_opts = sharding.ShardOptions(
        shard_index=shard_index, shard_count=shard_count, drop_remainder=True
    )
    source_obj = RangeDataSource(start=0, stop=24, step=1)
    sampler_obj = samplers.SequentialSampler(
        num_records=24, shard_options=shard_opts
    )
    loader = data_loader_lib.DataLoader(
        data_source=source_obj,
        sampler=sampler_obj,
        worker_count=worker_count,
        read_options=self.read_options,
    )
    full_expected = list(loader)
    itr = iter(loader)
    try:
      prefix = [next(itr) for _ in range(steps)]
      raw_state = itr.get_state()
      actual_dict = json.loads(raw_state.decode())
      parent_state = itr._iterator._parent.get_state()
    finally:
      itr._iterator.close()

    if "iterators_in_use_states" not in parent_state:
      next_index_in_cycle = 0
      workers_state = [parent_state]
    else:
      next_index_in_cycle = parent_state["next_index_in_cycle"]
      workers_state = parent_state["iterators_in_use_states"]
    w_count = len(workers_state)
    global_w_count = w_count * shard_count
    local_offset = shard_index - global_w_count
    expected_dict = {
        "version": 2,
        "last_seen_indices": {
            str(i): int(
                local_offset
                + i * shard_count
                + workers_state[i]["next_index"] * global_w_count
            )
            for i in range(w_count)
        },
        "last_worker_index": next_index_in_cycle - 1,
        "worker_count": worker_count,
        "sampler": repr(sampler_obj),
        "data_source": repr(source_obj),
    }
    self.assertEqual(actual_dict, expected_dict)
    self.assertEqual(raw_state, json.dumps(expected_dict, indent=4).encode())

    restored_loader = data_loader_lib.DataLoader(
        data_source=source_obj,
        sampler=sampler_obj,
        worker_count=worker_count,
        read_options=self.read_options,
    )
    restored_itr = iter(restored_loader)
    restored_itr.set_state(raw_state)
    self.assertEqual(prefix + list(restored_itr), full_expected)


if __name__ == "__main__":
  absltest.main()
