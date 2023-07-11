# Copyright 2022 Google LLC
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
      logging.info("Enabeling shared memory.")

    self.data_source = data_source
    self.sampler = sampler
    self.operations = operations

    self._read_options = read_options or options.ReadOptions()
    self._multiprocessing_options = options.MultiprocessingOptions(
        num_workers=worker_count,
        per_worker_buffer_size=worker_buffer_size,
        enable_profiling=enable_profiling,
    )

  @property
  def multiprocessing_options(self) -> options.MultiprocessingOptions:
    return self._multiprocessing_options

  @property
  def worker_count(self) -> int:
    return self._multiprocessing_options.num_workers

  def __iter__(self) -> "PyGrainDatasetIterator":
    if self.multiprocessing_options.num_workers == 0:
      logging.info("DataLoader Uses SingleProcessIterator.")
      return PyGrainDatasetIterator(
          self,
          iterator_factory=lambda state: _SingleProcessIterator(self, state),
      )
    else:
      logging.info("DataLoader Uses MultiProcessorIterator.")
      return PyGrainDatasetIterator(
          self,
          iterator_factory=lambda state: _MultiProcessorIterator(self, state),
      )


def _create_iterator(
    iterator: contextlib.AbstractContextManager[Iterator[Any]],
) -> Iterator[Any]:
  with iterator as it:
    yield from it


class PyGrainDatasetIterator(collections.abc.Iterator):
  """DataLoader iterator providing get/set state functionality.

  This is the only iterator we expose to users. It wraps underlying
  _SingleProcessIterator/_MultipleProcessIterator. In order to set state,
  it recreates the underlying iterator fresh with a new state.
  """

  def __init__(self, data_loader: DataLoader, iterator_factory):
    self._data_loader = data_loader
    self._iterator_factory = iterator_factory

    # Responsible for providing iterator state
    self._iterator_for_state = iterator_factory(None)

    # Iterator yielding actual elements.
    self._iterator_yielding_elements = _create_iterator(
        self._iterator_for_state
    )

  def __iter__(self):
    return self

  def __next__(self):
    return next(self._iterator_yielding_elements)

  def get_state(self) -> bytes:
    # Copy state, so that further iterations don't affect state given to user.
    copied_state = copy.deepcopy(self._iterator_for_state.state)
    return json.dumps(copied_state, indent=4).encode()

  def set_state(self, state: bytes):
    """Sets the state for the undelrying iterator.

    Note that state is an implementation detail and can change in the future.
    Args:
      state: state to restore the underlying iterator to.
    """
    state = json.loads(state.decode())
    self._validate_state(state)
    self._iterator_for_state = self._iterator_factory(state)
    self._iterator_yielding_elements = _create_iterator(
        self._iterator_for_state
    )

  def _validate_state(self, state):
    """Validates that loaded state matches data loader definition."""
    # state can be None if Iterator never progressed before checkpointing.
    if state is None:
      return
    expected_num_workers = self._data_loader.multiprocessing_options.num_workers
    if state[_WORKER_COUNT] != expected_num_workers:
      raise ValueError(
          "Worker count in checkpoint does not match dataloader worker count.\n"
          f"worker count in checkpoint: {state[_WORKER_COUNT]}\n"
          f"worker count in dataloader: {expected_num_workers}"
      )

    if state[_SAMPLER] != repr(self._data_loader.sampler):
      raise ValueError(
          "Sampler in checkpoint does not match dataloader sampler.\n"
          f"sampler in checkpoint: {state[_SAMPLER]}\n"
          f"sampler in dataloader: {repr(self._data_loader.sampler)}"
      )

    if state[_DATA_SOURCE] != repr(self._data_loader.data_source):
      raise ValueError(
          "DataSource in checkpoint does not match datasource in dataloader.\n"
          f"data source in checkpoint: {state[_DATA_SOURCE]}\n"
          f"data source in dataloader: {repr(self._data_loader.data_source)}"
      )


class _SingleProcessIterator(collections.abc.Iterator):
  """Iterator that runs the data transformations in the main process.

  Please note that the checkpointing state of a _SingleProcessIterator is the
  same as that of a _MultiProcessorIterator with one process.
  """

  def __init__(self, data_loader: DataLoader, state=None):
    self._data_loader = data_loader
    self._process_idx = 0
    self.state = state
    if state is not None:
      data_loader.sampler.reset(
          state[_LAST_SEEN_INDICES].get(str(self._process_idx))
      )
    else:
      data_loader.sampler.reset(-1)
    self._input_iterator = _read_and_transform_data(
        data_loader.sampler,
        data_loader.data_source,
        data_loader.operations,
        data_loader._read_options,  # pylint: disable=protected-access
    )

  def __next__(self):
    next_element = next(self._input_iterator)
    if self.state is None:
      self.state = {
          _VERSION: _CHECKPOINT_VERSION_NUMBER,
          _LAST_SEEN_INDICES: {},
          _WORKER_COUNT: self._data_loader.worker_count,
          _SAMPLER: repr(self._data_loader.sampler),
          _DATA_SOURCE: repr(self._data_loader.data_source),
      }
    self.state[_LAST_SEEN_INDICES][
        str(self._process_idx)
    ] = next_element.metadata.index
    self.state[_LAST_WORKER_INDEX] = self._process_idx
    return next_element.data

  def __iter__(self):
    return self

  def __enter__(self) -> _SingleProcessIterator:
    logging.debug("__enter__ of single process iterator is called.")
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    logging.debug("__exit__ of single process iterator is called.")
    self.state = None


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


# "_read_and_transform_data" as well as "_read_data" need to be top level
# functions in order to be pickle-able (needed for multiprocessing.)
def _read_and_transform_data(
    sampler: Sampler,
    data_source: RandomAccessDataSource,
    operations: Sequence[transforms.Transformation | Operation],
    read_options: options.ReadOptions,
):
  """Reads input data and applies operations to it."""
  iterator = _read_data(sampler, data_source, read_options=read_options)
  for operation in operations:
    iterator = _apply_transform(operation, iterator)
  return iterator


def _read_data(
    sampler: Sampler,
    data_source: RandomAccessDataSource,
    read_options: options.ReadOptions,
):
  """Reads sampled record indices from the data source and yields records."""
  num_prefetch = read_options.prefetch_buffer_size
  thread_pool = futures.ThreadPoolExecutor(read_options.num_threads)

  def prefetch_element(metadata: record.RecordMetadata) -> record.Record:
    data = data_source[metadata.record_key]
    return record.Record(metadata=metadata, data=data)  # pytype: disable=unsupported-operands

  with use_context_if_available(data_source):
    # Fill buffer.
    buffer = collections.deque()
    while len(buffer) < num_prefetch:
      try:
        metadata = next(sampler)
      except StopIteration:
        break
      buffer.append(thread_pool.submit(prefetch_element, metadata))
    # Iterate until buffer is empty. We try to add a new element each time we
    # remove one element.
    while buffer:
      element = buffer.popleft().result()
      yield element
      try:
        metadata = next(sampler)
      except StopIteration:
        continue
      buffer.append(thread_pool.submit(prefetch_element, metadata))


class GrainPoolProcessingError(Exception):
  """Raised when input processing in Grain Pool fails."""


class MultiProcessorIteratorInvalidStateError(Exception):
  """Raised when iterator is an invalid state and can't be iterated on."""


class _MultiProcessorIterator(collections.abc.Iterator):
  """Iterator that runs the data transformations in separate child processes.

  Note: MultiProcessorIterator implements the Context Manager protocol to clean
  resources. As such, it must be created using the "with" statement.

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

  def __init__(self, data_loader: DataLoader, state=None):
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
    self.state = state
    # If state doesn't contain indices for all workers, then it means one worker
    # didn't produce any output batches before checkpoint and thus we start from
    # the very beginning.
    if (
        self.state is not None
        and len(state[_LAST_SEEN_INDICES]) == data_loader.worker_count
    ):
      min_last_seen_index = min(state[_LAST_SEEN_INDICES].values())
      self._data_loader.sampler.reset(min_last_seen_index)
    else:
      self._data_loader.sampler.reset(-1)

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
            self.state,
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

    # GrainPool is a generic interface, that takes a function that only accepts
    # an input iterable. Our function "read_and_transform_data" accepts the
    # sampler (the iterable) as well as "data_source" and "operations", thus we
    # create a partial function out of it after fixing the "data_source" and
    # the "operations" arguments.
    transformation_function = functools.partial(
        _read_and_transform_data,
        data_source=data_loader.data_source,
        operations=data_loader.operations,
        read_options=data_loader._read_options,  # pylint: disable=protected-access
    )

    def discard_element(
        record_metadata: record.RecordMetadata, worker_idx: int
    ) -> bool:
      if state is None:
        return False
      # json.loads converts int keys to str, thus we convert worker_idx to str.
      last_seen_index_at_worker = state[_LAST_SEEN_INDICES].get(str(worker_idx))
      if last_seen_index_at_worker is None:
        return False
      return record_metadata.index <= last_seen_index_at_worker

    exception_from_pool = None
    worker_idx_to_start_processing = 0
    worker_idx_to_start_reading = 0
    if state is not None:
      if len(state[_LAST_SEEN_INDICES]) == data_loader.worker_count:
        min_last_seen_indices = min(state[_LAST_SEEN_INDICES].values())
        min_last_seen_indices_worker = (
            min_last_seen_indices % data_loader.worker_count
        )
        worker_idx_to_start_processing = (
            min_last_seen_indices_worker + 1
        ) % data_loader.worker_count

      worker_idx_to_start_reading = (
          state[_LAST_WORKER_INDEX] + 1
      ) % data_loader.worker_count

    def read_thread_should_stop():
      return (
          termination_event.is_set() or not threading.main_thread().is_alive()
      )

    mp_options = data_loader.multiprocessing_options
    with grain_pool.GrainPool(
        ctx=ctx,
        elements_to_process=data_loader.sampler,
        transformation_function=transformation_function,
        num_processes=mp_options.num_workers,
        elements_to_buffer_per_process=mp_options.per_worker_buffer_size,
        enable_profiling=mp_options.enable_profiling,
        discard_element_function=discard_element,
        worker_idx_to_start_processing=worker_idx_to_start_processing,
        worker_idx_to_start_reading=worker_idx_to_start_reading,
    ) as g_pool:
      try:
        for element in g_pool:
          if read_thread_should_stop():
            break
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
                  element.worker_idx,
              ),
              reader_queue,
              read_thread_should_stop,
          )
      # This exception could arise from user-provide code. Propagating it to
      # the main thread to re-raise it as is.
      except Exception as e:  # pylint: disable=broad-except
        exception_from_pool = e

      if exception_from_pool is not None:
        multiprocessing_common.add_element_to_queue(
            exception_from_pool, reader_queue, read_thread_should_stop
        )
      else:
        multiprocessing_common.add_element_to_queue(
            _GrainPoolProcessingComplete(),
            reader_queue,
            read_thread_should_stop,
        )
    logging.info("DataLoader reader thread completed!")

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

  def _update_state(self, last_seen_index: int, worker_idx: int):
    if self.state is None:
      self.state = {
          _VERSION: _CHECKPOINT_VERSION_NUMBER,
          _LAST_SEEN_INDICES: {},
          _WORKER_COUNT: self._data_loader.worker_count,
          _SAMPLER: repr(self._data_loader.sampler),
          _DATA_SOURCE: repr(self._data_loader.data_source),
      }
    # json.loads converts int keys to string, thus we convert worker_idx to str.
    self.state[_LAST_SEEN_INDICES][str(worker_idx)] = last_seen_index
    self.state[_LAST_WORKER_INDEX] = worker_idx

  def __next__(self):
    if not self._can_iterate():
      raise MultiProcessorIteratorInvalidStateError(
          "MultiProcessorIterator is in an invalid state. Note that"
          " MultiProcessorIterator should be used with a 'with' statement."
      )
    reader_queue_element = multiprocessing_common.get_element_from_queue(
        self._reader_queue, self._termination_event.is_set  # pytype: disable=attribute-error
    )
    if isinstance(reader_queue_element, Exception):
      raise GrainPoolProcessingError() from reader_queue_element
    if (
        reader_queue_element == _GRAIN_POOL_PROCESSING_COMPLETE
        or reader_queue_element == multiprocessing_common.SYSTEM_TERMINATED
    ):
      raise StopIteration
    # None is OK here, as it's what the user-provided input pipeline produces.

    # pytype:disable=attribute-error
    result = multiprocessing_common.get_async_result(
        reader_queue_element.async_result, self._termination_event.is_set
    )
    self._update_state(
        reader_queue_element.max_element_index,
        reader_queue_element.worker_index,
    )
    # pytype:enable=attribute-error
    return result
