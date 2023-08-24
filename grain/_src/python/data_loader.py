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
import copy
import dataclasses
import functools
import json
from multiprocessing import pool
from multiprocessing import shared_memory
import os
import queue
import threading
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

from absl import logging
from concurrent import futures
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import usage_logging
import multiprocessing as mp
from grain._src.python import grain_pool
from grain._src.python import multiprocessing_common
from grain._src.python import options
from grain._src.python import record
from grain._src.python.data_sources import RandomAccessDataSource
from grain._src.python.operations import BatchOperation
from grain._src.python.operations import Operation
from grain._src.python.operations import SharedMemoryMetadata
from grain._src.python.samplers import Sampler
import numpy as np
import tree


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


@dataclasses.dataclass(frozen=True, slots=True)
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
_QueueElement = _ReaderQueueElement | _GrainPoolProcessingComplete | Exception


@contextlib.contextmanager
def use_context_if_available(obj):
  """Uses with statement if obj is a context manager, else just uses the object."""
  if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
    with obj:
      yield
  else:
    yield


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

    # Shared memory should be enabled in Batch operation iff worker_count > 0.
    if (
        worker_count > 0
        and len(operations)
        and isinstance(operations[-1], BatchOperation)
    ):
      operations[-1]._enable_shared_memory()
      logging.info("Enabling shared memory.")

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

  def __iter__(self) -> PyGrainDatasetIterator:
    return PyGrainDatasetIterator(self, self._create_initial_state())

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
        _DATA_SOURCE: repr(self._data_source),
    }

  def _read_data(self, last_seen_index: int) -> Iterator[record.Record]:
    """Reads sampled record indices from the data source and yields records."""
    # We use a thread pool to read elements and add them to a buffer in the
    # background.
    # The main thread simply gets elements from the buffer and waits for them
    # to be available.
    next_index = last_seen_index + self._global_num_workers

    buffer = collections.deque()
    buffer_size = self._read_options.prefetch_buffer_size

    def prefetch_element(index: int) -> record.Record:
      metadata = self._sampler[index]
      data = self._data_source[metadata.record_key]
      return record.Record(metadata=metadata, data=data)

    with futures.ThreadPoolExecutor(self._read_options.num_threads) as executor:
      # Fill the buffer initially.
      while len(buffer) < buffer_size:
        buffer.append(executor.submit(prefetch_element, next_index))
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
        buffer.append(executor.submit(prefetch_element, next_index))
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
          f"sampler in dataloader: {repr(self._sampler)}"
      )
    if state[_DATA_SOURCE] != repr(self._data_source):
      raise ValueError(
          "DataSource in checkpoint does not match datasource in dataloader.\n"
          f"data source in checkpoint: {state[_DATA_SOURCE]}\n"
          f"data source in dataloader: {repr(self._data_source)}"
      )


def _iterator_with_context(
    iterator: contextlib.AbstractContextManager[Iterator[_T]],
) -> Iterator[_T]:
  with iterator as it:
    yield from it


class PyGrainDatasetIterator(collections.abc.Iterator[_T]):
  """DataLoader iterator providing get/set state functionality.

  This is the only iterator we expose to users. It wraps underlying
  _SingleProcessIterator/_MultipleProcessIterator. In order to set state,
  it recreates the underlying iterator fresh with a new state.
  """

  def __init__(self, data_loader: DataLoader, state: _IteratorState):
    self._data_loader = data_loader
    self._data_loader._validate_state(state)
    self._initial_iterator_state = state
    self._raw_iterator = None
    self._iterator = None

  def __iter__(self) -> PyGrainDatasetIterator[_T]:
    return self

  def __next__(self) -> _T:
    if self._iterator is None:
      if self._data_loader.multiprocessing_options.num_workers == 0:
        self._raw_iterator = _SingleProcessIterator(
            self._data_loader, self._initial_iterator_state
        )
        self._iterator = self._raw_iterator
      else:
        self._raw_iterator = _MultiProcessorIterator(
            self._data_loader, self._initial_iterator_state
        )
        self._iterator = _iterator_with_context(self._raw_iterator)
    return next(self._iterator)

  def get_state(self) -> bytes:
    if self._raw_iterator:
      state = self._raw_iterator.get_state()
    else:
      state = self._initial_iterator_state
    return json.dumps(state, indent=4).encode()

  def set_state(self, state: bytes):
    """Sets the state for the undelrying iterator.

    Note that state is an implementation detail and can change in the future.
    Args:
      state: state to restore the underlying iterator to.
    """
    state = json.loads(state.decode())
    self._data_loader._validate_state(state)  # pylint: disable=protected-access
    self._initial_iterator_state = state
    self._raw_iterator = None
    self._iterator = None

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
  match transform:
    case transforms.MapTransform():
      fn = lambda r: (record.Record(r.metadata, transform.map(r.data)), True)
    case transforms.RandomMapTransform():
      fn = lambda r: (
          record.Record(
              r.metadata, transform.random_map(r.data, r.metadata.rng)
          ),
          True,
      )
    case transforms.TfRandomMapTransform():
      fn = lambda r: (
          record.Record(
              r.metadata, transform.np_random_map(r.data, r.metadata.rng)
          ),
          True,
      )
    case transforms.FilterTransform():
      fn = lambda r: (r, bool(transform.filter(r.data)))
    case _:
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
      raise ValueError(
          f"PyGrain encountered an error when applying {transform}."
      ) from e
    if filter_result:
      yield output_record


class _SingleProcessIterator(collections.abc.Iterator):
  """Iterator that runs the data transformations in the main process.

  Please note that the checkpointing state of a _SingleProcessIterator is the
  same as that of a _MultiProcessorIterator with one process.
  """

  def __init__(self, data_loader: DataLoader, state: _IteratorState):
    self._data_loader = data_loader
    self._state = state
    self._iterator = self._data_loader._read_and_transform_data(
        self._state[_LAST_SEEN_INDICES]["0"]
    )

  def __next__(self):
    next_element = next(self._iterator)
    self._state[_LAST_SEEN_INDICES]["0"] = next_element.metadata.index
    return next_element.data

  def get_state(self) -> _IteratorState:
    return copy.deepcopy(self._state)

  def __iter__(self):
    return self


class GrainPoolProcessingError(Exception):
  """Raised when input processing in Grain Pool fails."""


class MultiProcessorIteratorInvalidStateError(Exception):
  """Raised when iterator is an invalid state and can't be iterated on."""


class _MultiProcessorIterator(collections.abc.Iterator):
  """Iterator that runs the data transformations in separate child processes.

  Note: MultiProcessorIterator implements the Context Manager protocol to clean
  resources. As such, it must be used within a "with" statement.

  Checkpointing for _MultiProcessorIterator:
  _MultiProcessorIterator uses GrainPool, which distributes RecordMetadata from
  the Sampler among worker processes in a round robin fashion. Due to Filter (or
  packing) operations, some workers can process more elements than others at a
  given training step. Checkpointing logic goes as follows:
  1) With each output batch produced, GrainPool emits the worker_index of The
     worker that processed the batch.
  2) _MultiProcessorIterator keeps track of the last_seen_index at each worker.
  3) When restoring from a state, _MultiProcessorIterator checks what is the
     minumum last_seen_index (among the last seen indices for all workers.) and
     which worker processed that index.
  4) The Sampler is reset to start from the next index (to the index from #3)
     and GrainPool is instructed to start distributing indices to the next
     worker (to the one that processed the index from #3).
  5) To avoid processing elements twice, we pass a discard_function to the pool,
     which discards any element that was already processed before (where element
     index is <= to the last_seen_index) at a given worker.
  6) GrainPool keeps track of which worker produced the last output batch handed
     to the user. Upon restarting from a checkpoint, round robin reading is
     started from the next worker (to the worker producing the last batch).
  """

  def __init__(self, data_loader: DataLoader, state: _IteratorState):
    """Creates an iterator for given data_loader that uses multiple processes.

    Args:
      data_loader: data_loader for which the iterator is created.
      state: initial state to load iterator from.
    """
    self._data_loader = data_loader
    self._reader_queue = None
    self._reader_thread_pool = None
    self._termination_event = None
    self._reader_thread = None
    self._state = state

  def __enter__(self):
    mp_options = self._data_loader.multiprocessing_options
    max_buffered_elements = (
        mp_options.num_workers * mp_options.per_worker_buffer_size
    )
    self._reader_queue = queue.Queue(maxsize=max_buffered_elements)
    self._reader_thread_pool = pool.ThreadPool(max_buffered_elements)
    self._termination_event = threading.Event()
    self._reader_thread = threading.Thread(
        target=_MultiProcessorIterator._process_elements,
        args=(
            self._data_loader,
            self._reader_queue,
            self._reader_thread_pool,
            self._termination_event,
            self._state,
        ),
    )
    self._reader_thread.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    # pytype: disable=attribute-error
    self._termination_event.set()
    self._reader_thread_pool.close()
    self._reader_thread.join()
    self._reader_thread_pool.join()
    # pytype: enable=attribute-error
    self._termination_event = None
    self._reader_thread_pool = None
    self._reader_thread = None
    self._reader_queue = None

  @staticmethod
  def _read_and_unlink_shared_memory(element: Any) -> Any:
    """Reads and unlinks shared memory blocks if element has any."""
    if isinstance(element, SharedMemoryMetadata):
      shm = shared_memory.SharedMemory(name=element.name)
      array_in_shm = np.ndarray(
          element.shape, dtype=element.dtype, buffer=shm.buf
      )
      # The Numpy array is to be handed to the user, who might use it for an
      # indefinite time or pass it to other processes. We cannot keep the shared
      # memory block indefinitely, thus chose to copy data out of it here.
      data_copied_from_shm = array_in_shm.copy()
      # make sure to free the shared memory block.
      shm.close()
      shm.unlink()
      return data_copied_from_shm
    else:
      return element

  @staticmethod
  def _process_elements(
      data_loader: DataLoader,
      reader_queue: queue.Queue[_QueueElement],
      thread_pool: pool.ThreadPool,
      termination_event: threading.Event,
      state: _IteratorState,
  ) -> None:
    """Processes elements read from grain pool asynchronously."""
    ctx = mp.get_context("spawn")

    worker_index_to_start_reading = (
        state[_LAST_WORKER_INDEX] + 1
    ) % data_loader.multiprocessing_options.num_workers

    def read_thread_should_stop():
      return (
          termination_event.is_set() or not threading.main_thread().is_alive()
      )

    def get_element_producer_fn(worker_index: int, worker_count: int):
      del worker_count
      last_seen_index = state[_LAST_SEEN_INDICES].get(str(worker_index))
      iterator = data_loader._read_and_transform_data(last_seen_index)  # pylint: disable=protected-access
      yield from iterator

    try:
      with grain_pool.GrainPool(
          ctx=ctx,
          get_element_producer_fn=get_element_producer_fn,
          worker_index_to_start_reading=worker_index_to_start_reading,
          options=data_loader.multiprocessing_options,
      ) as g_pool:
        for element in g_pool:
          if read_thread_should_stop():
            break
          # Note: Do we really need this thread pool? Can we return a weak
          # ref instead of unlinking from memory?
          # This method is already running in a background thread of the main
          # process.
          async_result = thread_pool.apply_async(
              tree.map_structure,
              args=(
                  _MultiProcessorIterator._read_and_unlink_shared_memory,
                  element.record.data,
              ),
          )
          multiprocessing_common.add_element_to_queue(
              _ReaderQueueElement(
                  async_result,
                  element.record.metadata.index,
                  element.worker_index,
              ),
              reader_queue,
              read_thread_should_stop,
          )
    # This exception could arise from user-provide code. Propagating it to
    # the main thread to re-raise it as is.
    except Exception as e:  # pylint: disable=broad-except
      multiprocessing_common.add_element_to_queue(
          e, reader_queue, read_thread_should_stop
      )
      return
    multiprocessing_common.add_element_to_queue(
        _GrainPoolProcessingComplete(),
        reader_queue,
        read_thread_should_stop,
    )

  def _can_iterate(self):
    """Checks whether the object is in a state where it can be iterated on."""
    return (
        self._reader_queue is not None
        and self._termination_event is not None
        and self._reader_thread_pool is not None
        and self._reader_thread is not None
    )

  def __iter__(self):
    if not self._can_iterate():
      raise MultiProcessorIteratorInvalidStateError(
          "MultiProcessorIterator is in an invalid state. Note that"
          " MultiProcessorIterator should be used with a 'with' statement."
      )
    return self

  def get_state(self) -> _IteratorState:
    return copy.deepcopy(self._state)

  def __next__(self):
    if not self._can_iterate():
      raise MultiProcessorIteratorInvalidStateError(
          "MultiProcessorIterator is in an invalid state. Note that"
          " MultiProcessorIterator should be used with a 'with' statement."
      )
    element = multiprocessing_common.get_element_from_queue(
        self._reader_queue, self._termination_event.is_set  # pytype: disable=attribute-error
    )
    if isinstance(element, Exception):
      raise GrainPoolProcessingError() from element
    if (
        element == _GRAIN_POOL_PROCESSING_COMPLETE
        or element == multiprocessing_common.SYSTEM_TERMINATED
    ):
      raise StopIteration

    if not isinstance(element, _ReaderQueueElement):
      raise ValueError(
          f"Got invalid element type from GrainPool: {type(element)}"
      )

    result = multiprocessing_common.get_async_result(
        element.async_result, self._termination_event.is_set
    )
    if isinstance(result, multiprocessing_common._SystemTerminated):  # pylint: disable=protected-access
      raise StopIteration
    self._state[_LAST_SEEN_INDICES][
        str(element.worker_index)
    ] = element.max_element_index
    self._state[_LAST_WORKER_INDEX] = element.worker_index
    return result
