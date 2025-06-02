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
"""DataLoader is reponsible for loading and transforming input data."""

from __future__ import annotations

import collections
from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import json
from multiprocessing import pool
from multiprocessing import queues
import os
import sys
import time
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple, TypeVar, Union

from absl import logging
from etils import epath
from concurrent import futures
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import tree_lib
from grain._src.core import usage_logging
import multiprocessing as mp
from grain._src.python import checkpointing
from grain._src.python import grain_pool
from grain._src.python import options
from grain._src.python import record
from grain._src.python.data_sources import RandomAccessDataSource
from grain._src.python.operations import BatchOperation
from grain._src.python.operations import Operation
from grain._src.python.samplers import Sampler
from grain._src.python.shared_memory_array import SharedMemoryArray
import numpy as np

from grain._src.core import monitoring

_api_usage_counter = monitoring.Counter(
    "/grain/python/data_loader/api",
    monitoring.Metadata(description="API initialization counter."),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)
_iterator_get_next_metric = monitoring.EventMetric(
    "/grain/python/data_loader/iterator_get_next",
    monitoring.Metadata(
        description="Gauge for DataLoaderIterator.__next__() latency.",
        units=monitoring.Units.NANOSECONDS,
    ),
    root=grain_monitoring.get_monitoring_root(),
)

_T = TypeVar("_T")
_IteratorState = dict[str, Any]

# Dictionary keys used in checkpoints.
_VERSION = "version"
_LAST_SEEN_INDICES = "last_seen_indices"
_LAST_WORKER_INDEX = "last_worker_index"
_WORKER_COUNT = "worker_count"
_SAMPLER = "sampler"
_DATA_SOURCE = "data_source"
# Version of current checkpoint format.
# Version 1 was experimental and is no longer supported.
_CHECKPOINT_VERSION_NUMBER = 2


def _validate_operations(operations: Sequence[Operation]) -> None:
  """Validates user-provided operations."""
  for operation, operation_idx in enumerate(operations):
    if (
        isinstance(operation, BatchOperation)
        and operation_idx < len(operations) - 1
    ):
      raise ValueError(
          "Batch Operation is only allowed at the end of the "
          "pipeline. Found a Batch Operation at position "
          f"{operation_idx} in the input operations."
      )


def _determine_worker_count(input_worker_count: int | None) -> int:
  """Determines count of child processes to use."""
  if input_worker_count is not None:
    return int(input_worker_count)
  if (os_cpu_count := os.cpu_count()) is not None:
    return os_cpu_count
  else:
    raise ValueError("Can't determine worker count. Please set worker count.")


@dataclasses.dataclass(slots=True, frozen=True)
class _ReaderQueueElement:
  """Element to be added to the reader queue."""

  async_result: pool.AsyncResult[Any]
  # max record index seen so far at worker with worker_index
  max_element_index: int
  # index of worker producing the element in [0, worker_count]
  worker_index: int


@dataclasses.dataclass(frozen=True)
class _GrainPoolProcessingComplete:
  """Indicates processing of grain pool is complete."""


_GRAIN_POOL_PROCESSING_COMPLETE = _GrainPoolProcessingComplete()
_QueueElement = Union[
    _ReaderQueueElement, _GrainPoolProcessingComplete, Exception
]


@contextlib.contextmanager
def use_context_if_available(obj):
  """Uses with statement if obj is a context manager, else just uses the object."""
  if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
    with obj:
      yield
  else:
    yield


@dataclasses.dataclass
class CopyNumPyArrayToSharedMemory(transforms.MapTransform):
  """If `element` contains NumPy array copy it to SharedMemoryArray."""

  def map(self, element: Any) -> Any:
    def copy_if_applied(element: Any) -> Any:
      if (
          not isinstance(element, np.ndarray)
          or element.dtype.hasobject
          or not element.flags.c_contiguous
      ):
        return element

      shared_memory_arr = SharedMemoryArray(element.shape, element.dtype)
      np.copyto(shared_memory_arr, element, casting="no")
      return shared_memory_arr.metadata

    return tree_lib.map_structure(copy_if_applied, element)


class DataLoader:
  """DataLoader loads and transforms input data."""

  def __init__(
      self,
      *,
      data_source: RandomAccessDataSource,
      sampler: Sampler,
      operations: Sequence[transforms.Transformation | Operation] = (),
      worker_count: Optional[int] = 0,
      worker_buffer_size: int = 1,
      shard_options: sharding.ShardOptions | None = None,
      read_options: options.ReadOptions | None = None,
      enable_profiling: bool = False,
  ):
    """Loads and transforms input data.

    Args:
      data_source: Responsible for retrieving individual records based on their
        indices.
      sampler: Sampler is responsible for providing the index of the next record
        to read and transform.
      operations: Sequence of operations (e.g. Map, Filter) applied to the data.
      worker_count: Number of child processes launched to parallelize the
        transformations among. Zero means processing runs in the same process.
        None lets the python backend choose the value.
      worker_buffer_size: Count of output batches to produce in advance per
        worker. This ensures batches are ready when the consumer requests them.
      shard_options: Options for how data should be sharded when using multiple
        machines (~ JAX processes) and data parallelism.
      read_options: Options to use for reading. See ReadOptions.
      enable_profiling: If True, profiling info is logged. Note, it only
        supports worker_count >= 1 at the moment.
    """
    usage_logging.log_event("PyGrainDataLoader", tag_3="PyGrain")
    _api_usage_counter.Increment("DataLoader")
    if worker_count and worker_count < 0:
      raise ValueError(
          "Worker count should be greater than or equal zero."
          f"Current worker_count is {worker_count}."
      )
    if worker_buffer_size < 0:
      raise ValueError(
          "Worker buffer size must be greater than or equal zero."
          f"Current worker_buffer_size is {worker_buffer_size}."
      )

    worker_count = _determine_worker_count(worker_count)
    if worker_count > 0:

      # Shared memory should be enabled iff worker_count > 0.
      if operations and isinstance(operations[-1], BatchOperation):
        logging.info("Enabling SharedMemoryArray for BatchOperation.")
        operations[-1]._enable_shared_memory()
      else:
        logging.info("Adding CopyNumPyArrayToSharedMemory MapTransform.")
        operations = list(operations) + [CopyNumPyArrayToSharedMemory()]

    self._data_source = data_source
    self._sampler = sampler
    self._operations = operations

    self._read_options = read_options or options.ReadOptions()
    self._multiprocessing_options = options.MultiprocessingOptions(
        num_workers=worker_count,
        per_worker_buffer_size=worker_buffer_size,
        enable_profiling=enable_profiling,
    )
    self._shard_options = shard_options
    if self._shard_options is None:
      # Previously the Sampler owned the sharding. Try to get sharding from the
      # sampler.
      # pylint: disable=protected-access
      if hasattr(self._sampler, "_shard_options") and isinstance(
          self._sampler._shard_options, sharding.ShardOptions
      ):
        self._shard_options = self._sampler._shard_options
      else:
        raise ValueError(
            "No shard options were provided to the DataLoader. Please pass "
            "shard options to the DataLoader. Previously sharding was handled "
            "by the Sampler but going forward sharding will be handled by the "
            "DataLoader for greater flexibility."
        )
      # pylint: enable=protected-access

  @property
  def multiprocessing_options(self) -> options.MultiprocessingOptions:
    return self._multiprocessing_options

  @functools.cached_property
  def _local_num_workers(self):
    """Returns the number of workers across (within this data shard)."""
    return max(self._multiprocessing_options.num_workers, 1)

  @functools.cached_property
  def _global_num_workers(self):
    """Returns the number of workers across all data shards."""
    return self._local_num_workers * self._shard_options.shard_count  # pytype: disable=attribute-error

  def __iter__(self) -> DataLoaderIterator:
    return DataLoaderIterator(self, self._create_initial_state())

  def _create_initial_state(self) -> _IteratorState:
    """Create the initial state for checkpoints."""
    # We have `shard_count` machines iterating over the sampler, each machine
    # uses `num_workers` workers. We avoid a global or local (=within machine)
    # queue distributing indices among the workers. Such queue could easily
    # become the bottleneck. Instead, we evenly iterate over the sampler, each
    # worker starts at a `local_offset` and does `global_num_workers` steps.
    # The `last_seen_indices` are negative because we start reading at the
    # next_index=last_seen_index+global_num_workers.
    # Example:
    # For shard_count=3 (usually 3 JAX processes) and num_workers=5 we get the
    # following:
    # shard_index=0 gets values [-15, -12, -9, -6, -3]
    # shard_index=1 gets values [-14, -11, -8, -5, -2]
    # shard_index=2 gets values [-13, -10, -7, -4, -1]
    # The corresponding first indices read by the workers will be (since the
    # global number of workers is 3*5=15):
    # shard_index=0 gets values [0, 3, 6, 9, 12]
    # shard_index=1 gets values [1, 4, 7, 10, 13]
    # shard_index=2 gets values [2, 5, 8, 11, 14]
    local_offset = self._shard_options.shard_index - self._global_num_workers  # pytype: disable=attribute-error
    last_seen_indices = {
        str(i): local_offset + i * self._shard_options.shard_count  # pytype: disable=attribute-error
        for i in range(self._local_num_workers)
    }
    return {
        _VERSION: _CHECKPOINT_VERSION_NUMBER,
        _LAST_SEEN_INDICES: last_seen_indices,
        _LAST_WORKER_INDEX: -1,
        _WORKER_COUNT: self._multiprocessing_options.num_workers,
        _SAMPLER: repr(self._sampler),
        _DATA_SOURCE: _source_repr(self._data_source),
    }

  def _read_data(self, last_seen_index: int) -> Iterator[record.Record]:
    """Reads sampled record indices from the data source and yields records."""
    # We use a thread pool to read elements and add them to a buffer in the
    # background.
    # The main thread simply gets elements from the buffer and waits for them
    # to be available.
    next_index = last_seen_index + self._global_num_workers

    def fetch_element(index: int) -> record.Record:
      metadata = self._sampler[index]
      data = self._data_source[metadata.record_key]
      return record.Record(metadata=metadata, data=data)

    if self._read_options.num_threads == 0:
      while True:
        try:
          element = fetch_element(next_index)
        except IndexError:
          return
        yield element
        next_index += self._global_num_workers

    buffer = collections.deque()
    buffer_size = self._read_options.prefetch_buffer_size

    with futures.ThreadPoolExecutor(self._read_options.num_threads) as executor:
      # Fill the buffer initially.
      while len(buffer) < buffer_size:
        buffer.append(executor.submit(fetch_element, next_index))
        next_index += self._global_num_workers

      # Iterate until we get an IndexError. The IndexError indicates that we
      # reached the end of the Sampler.
      while True:
        try:
          element = buffer.popleft().result()
        except IndexError:
          # End of sampler.
          return
        yield element
        buffer.append(executor.submit(fetch_element, next_index))
        next_index += self._global_num_workers

  def _read_and_transform_data(
      self, last_seen_index: int
  ) -> Iterator[record.Record]:
    """Reads input data and applies operations to it."""
    iterator = self._read_data(last_seen_index)
    for operation in self._operations:
      iterator = _apply_transform(operation, iterator)
    return iterator

  def _validate_state(self, state: _IteratorState):
    """Validates that loaded state matches data loader definition."""
    # state can be None if Iterator never progressed before checkpointing.
    expected_worker_count = self._multiprocessing_options.num_workers
    if state[_WORKER_COUNT] != expected_worker_count:
      raise ValueError(
          "Worker count in checkpoint does not match dataloader worker count.\n"
          f"worker count in checkpoint: {state[_WORKER_COUNT]}\n"
          f"worker count in dataloader: {expected_worker_count}"
      )
    if state[_SAMPLER] != repr(self._sampler):
      raise ValueError(
          "Sampler in checkpoint does not match dataloader sampler.\n"
          f"sampler in checkpoint: {state[_SAMPLER]}\n"
          f"sampler in dataloader: {repr(self._sampler)}\n"
          "Grain uses `repr(sampler)` to validate the sampler, so you "
          "may need to implement a custom `__repr__`."
      )
    if state[_DATA_SOURCE] != _source_repr(self._data_source):
      raise ValueError(
          "DataSource in checkpoint does not match datasource in dataloader.\n"
          f"data source in checkpoint: {state[_DATA_SOURCE]}\n"
          f"data source in dataloader: {_source_repr(self._data_source)}\n"
          "Grain uses `repr(data_source)` to validate the source, so you "
          "may need to implement a custom `__repr__`."
      )


def _iterator_with_context(
    iterator: contextlib.AbstractContextManager[Iterator[_T]],
) -> Iterator[_T]:
  with iterator as it:
    yield from it


def _source_repr(source: RandomAccessDataSource) -> str:
  """Returns a string representation of the source."""
  # If the source has data in memory avoid printing the data itself.
  if isinstance(source, (list, tuple, np.ndarray)):
    return str(type(source))
  return repr(source)


class GetElementProducerFn(grain_pool.GetElementProducerFn):
  """Implements `grain_pool.GetElementProducerFn`."""

  def __init__(
      self,
      state: _IteratorState,
      read_and_transform_data: Callable[[int], Iterator[record.Record]],
  ):
    self._state = state
    self._read_and_transform_data = read_and_transform_data

  def __call__(
      self,
      *,
      worker_index: int,
      worker_count: int,
      stats_out_queue: queues.Queue | None = None,
  ) -> Iterator[record.Record]:
    del worker_count
    last_seen_index = self._state[_LAST_SEEN_INDICES].get(str(worker_index))
    yield from self._read_and_transform_data(last_seen_index)


class DataLoaderIterator(collections.abc.Iterator[_T]):
  """DataLoader iterator providing get/set state functionality.

  This is the only iterator we expose to users. It wraps underlying
  MultipleProcessIterator. In order to set state, it recreates the underlying
  iterator fresh with a new state.

  Checkpointing for DataLoaderIterator:
  DataLoaderIterator uses GrainPool, which distributes RecordMetadata from
  produced records among worker processes in a round robin fashion. Generally,
  some workers can process more elements than others at a given training step.
  Checkpointing logic goes as follows:
  1) With each output batch produced, GrainPool emits the worker_index of The
     worker that processed the batch.
  2) DataLoaderIterator keeps track of the last_seen_index at each worker.
  3) When restoring from a state, DataLoaderIterator checks what is the
     minimum last_seen_index (among the last seen indices for all workers.) and
     which worker processed that index. GrainPool is instructed to start
     distributing indices to the next worker.
  """

  def __init__(self, data_loader: DataLoader, state: _IteratorState):
    self._data_loader = data_loader
    self._data_loader._validate_state(state)
    self._state = state
    self._raw_iterator = None
    self._iterator = None

  def __iter__(self) -> DataLoaderIterator[_T]:
    return self

  def _create_iterator(self) -> None:
    """Creates the wrapped `MultiProcessIterator` or in-process iterator."""
    if self._data_loader.multiprocessing_options.num_workers == 0:
      # Pipeline is going to be executed in the main process.
      self._raw_iterator = self._data_loader._read_and_transform_data(  # pylint: disable=protected-access
          self._state[_LAST_SEEN_INDICES]["0"]
      )
      self._iterator = self._raw_iterator
    else:
      state = self._state
      # Custom DataLoader can avoid pickling the `self._data_loader` object here
      # by e.g. making `_read_and_transform_data` a property.
      read_and_transform_data = self._data_loader._read_and_transform_data  # pylint: disable=protected-access

      get_element_producer_fn = GetElementProducerFn(
          state, read_and_transform_data
      )

      worker_index_to_start_reading = (
          state[_LAST_WORKER_INDEX] + 1
      ) % self._data_loader.multiprocessing_options.num_workers

      self._raw_iterator = grain_pool.MultiProcessIterator(
          get_element_producer_fn,
          self._data_loader.multiprocessing_options,
          worker_index_to_start_reading,
      )
      self._iterator = _iterator_with_context(self._raw_iterator)

  def __next__(self) -> _T:
    start_time = time.time_ns()
    if self._iterator is None:
      self._create_iterator()

    result_record = next(self._iterator)

    if isinstance(self._raw_iterator, grain_pool.MultiProcessIterator):
      last_worker_index = self._raw_iterator.get_last_worker_index()
      self._state[_LAST_WORKER_INDEX] = last_worker_index
    else:
      last_worker_index = 0
    self._state[_LAST_SEEN_INDICES][
        str(last_worker_index)
    ] = result_record.metadata.index
    _iterator_get_next_metric.Record(time.time_ns() - start_time)
    return result_record.data

  def get_state(self) -> bytes:
    return json.dumps(self._state, indent=4).encode()

  def set_state(self, state: bytes):
    """Sets the state for the underlying iterator.

    Note that state is an implementation detail and can change in the future.
    Args:
      state: state to restore the underlying iterator to.
    """
    state = json.loads(state.decode())
    self._data_loader._validate_state(state)  # pylint: disable=protected-access
    self._state: _IteratorState = state
    self._raw_iterator = None
    self._iterator = None

  ### BEGIN Orbax checkpointing API.
  # See orbax.checkpoint.v1.handlers.StatefulCheckpointable for more details.
  # See https://orbax.readthedocs.io/en/latest/ for usage examples.

  async def save(
      self, directory: checkpointing.PathAwaitingCreation
  ) -> Awaitable[None]:
    """Saves the iterator state to a directory.

    The current state (`get_state`) is used for saving, so any updates to the
    state after returning from this method will not affect the saved checkpoint.

    Args:
      directory: A path in the process of being created. Must call
        await_creation before accessing the physical path.

    Returns:
      A coroutine that has not been awaited. This is called by Orbax in a
      background thread to perform I/O without blocking the main thread.
    """
    state = self.get_state().decode()
    return checkpointing.background_save(directory, state)

  async def load(self, directory: epath.Path) -> Awaitable[None]:
    """Loads the iterator state from a directory.

    The state may be loaded and set in a background thread. The main thread
    should not alter the state content while the load is in progress.

    Args:
      directory: The directory to load the state from.

    Returns:
      A coroutine that has not been awaited. This is called by Orbax in a
      background thread to perform I/O without blocking the main thread.
    """

    def set_state_fn(state: str):
      self.set_state(state.encode())

    return checkpointing.background_load(directory, set_state_fn)

  ### END Orbax checkpointing API.

  def __str__(self):
    return f"PyGrainDatasetIterator(state={self.get_state().decode()})"


def _apply_transform(
    transform: transforms.Transformation | Operation,
    input_iterator: Iterator[record.Record],
) -> Iterator[record.Record]:
  """Applies the `transform` to records in the iterator."""
  fn: Callable[[record.Record], Tuple[record.Record, bool]] = None
  # pylint: disable=g-long-lambda
  # pytype: disable=attribute-error
  if isinstance(transform, transforms.MapTransform):
    fn = lambda r: (record.Record(r.metadata, transform.map(r.data)), True)
  elif isinstance(transform, transforms.RandomMapTransform):
    fn = lambda r: (
        record.Record(r.metadata, transform.random_map(r.data, r.metadata.rng)),
        True,
    )
  elif isinstance(transform, transforms.TfRandomMapTransform):
    fn = lambda r: (
        record.Record(
            r.metadata, transform.np_random_map(r.data, r.metadata.rng)
        ),
        True,
    )
  elif isinstance(transform, transforms.Filter):
    fn = lambda r: (r, bool(transform.filter(r.data)))
  elif isinstance(transform, transforms.Batch):
    batch_op = BatchOperation(
        batch_size=transform.batch_size,
        drop_remainder=transform.drop_remainder,
    )
    batch_op.disable_deprecation_message()
    for r in batch_op(input_iterator):
      yield r
  else:
    # Transform is a legacy style operation and __call__() yield output
    # records.
    for r in transform(input_iterator):
      yield r
  # pytype: enable=attribute-error
  # pylint: enable=g-long-lambda

  for input_record in input_iterator:
    try:
      output_record, filter_result = fn(input_record)
    except Exception as e:
      if sys.version_info >= (3, 11):
        e.add_note(f"\nThe error occurred in {transform}.")
      raise e
    if filter_result:
      yield output_record
