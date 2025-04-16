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
"""This module provides a way to distribute processing across multiple workers.

In the context of Grain we use the term "process" similar to JAX, where usually
each machine runs one Python process (identified by `jax.process_index()`).
In Grain each "process" can create additional Python child processes that we
call "workers".

GrainPool manages a set of Python processes. It's similar to
`multiprocessing.Pool` but optimises communication between the processes to
enable high throughput data pipelines.
The GrainPool works as follows:
* Parent process launches a set of "num_workers" child processes.
* Each child process produces elements by reading data and transforming it. The
  resulting elements are added to a queue (each child process has its queue).
* Parent process reads data from the children queues in a strict round-robin
  fashion.

Shutdown logic considerations:
* Child processes are launched as Daemon processes. In case of (unexpected)
  parent termination, child processes will be terminated by OS.
* System uses a multiprocessing event ("termination_event") for termination.
  Parent and child processes continuously check if the "termination_event" and
  if set, they break from what they are doing.
* We never block indefinitely when calling get() or put() on a queue. This
  ensures parent and child processes continue to check the termination_event.

MultiProcessIterator wraps GrainPool adding lifecycle management, checkpointing
support and multithreaded elements read.
"""

from __future__ import annotations

from collections.abc import Iterator
import cProfile
import dataclasses
from multiprocessing import context
from multiprocessing import pool
from multiprocessing import queues
from multiprocessing import synchronize
import pstats
import queue
import sys
import threading
import traceback
from typing import Any, Callable, Protocol, TypeVar, Union, runtime_checkable

from absl import flags
from absl import logging
import cloudpickle
from grain._src.core import parallel
from grain._src.core import tree_lib
from grain._src.core.config import config
import multiprocessing as mp
from grain._src.python import grain_logging
from grain._src.python import multiprocessing_common
from grain._src.python import record
from grain._src.python import shared_memory_array
from grain._src.python.options import MultiprocessingOptions  # pylint: disable=g-importing-member

T = TypeVar("T")

# Maximum number of threads for starting and stopping processes.
_PROCESS_MANAGEMENT_MAX_THREADS = 64
_PROCESS_JOIN_TIMEOUT = 10
_QUEUE_WAIT_TIMEOUT = 1
# Input queues contain small structures (record metadata), thus they are safe
# to have a big size.
_INPUT_QUEUE_MAX_SIZE = 10000


@dataclasses.dataclass
class _ProcessingComplete:
  """Indicates child process finished processing."""


_PROCESSING_COMPLETE = _ProcessingComplete()


@dataclasses.dataclass(slots=True, frozen=True)
class GrainPoolElement:
  """Wrapper for output records emited by Grain Pool."""

  record: Any
  worker_index: Any


class RemoteWorkerError(Exception):
  """Grain worker exception that can be pickled and sent over a queue."""

  def __init__(self, error: str, worker_index: int):
    super().__init__(
        f"Grain worker {worker_index} failed with the following"
        f" error:\n\n{error}"
    )
    self._error = error
    self._worker_index = worker_index

  def __reduce__(self):
    # Note that during pickling the RemoteWorkerError loses __traceback__ and
    # __cause__ attributes because they are irrelevant. Only the original error
    # attributes are preserved in the string form.
    return RemoteWorkerError, (self._error, self._worker_index)


def _print_profile(preamble: str, profile: cProfile.Profile):
  """Prints output of cProfile, sorted by cumulative time."""
  print(preamble)
  stats = pstats.Stats(profile).sort_stats(pstats.SortKey.CUMULATIVE)
  stats.print_stats()


@runtime_checkable
class GetElementProducerFn(Protocol[T]):
  """A callable class able to generate elements with serialization support."""

  def __call__(
      self,
      *,
      worker_index: int,
      worker_count: int,
      stats_out_queue: queues.Queue | None = None,
  ) -> Iterator[T]:
    """Returns a generator of elements."""

  def serialize(self) -> bytes:
    """Serializes itself and the result will be used by `deserialize`.

    If a class inherits from this class, it should make sure `deserialize`
    is compatible with this `serialize` function.
    i.e. `GetElementProducerFn.deserialize(obj.serialize())` should return the
    same object as `obj: GetElementProducerFn`.

    Returns:
      a serialized string of myself.
    """
    return cloudpickle.dumps(self)

  @classmethod
  def deserialize(cls, serialized: bytes) -> GetElementProducerFn[T]:
    """Deserializes the result from `serialize`."""
    del cls

    obj = cloudpickle.loads(serialized)
    if not isinstance(obj, GetElementProducerFn):
      raise ValueError(
          "`serialized` should be deserialized into `GetElementProducerFn`."
      )

    return obj


def parse_debug_flags(debug_flags: dict[str, Any]):
  """Parses debug flags."""

  flags.FLAGS["grain_py_debug_mode"].present = True
  flags.FLAGS["grain_py_dataset_visualization_output_dir"].present = True
  config.update("py_debug_mode", debug_flags["grain_py_debug_mode"])
  config.update(
      "py_dataset_visualization_output_dir",
      debug_flags["grain_py_dataset_visualization_output_dir"],
  )


def _initialize_and_get_element_producer(
    args_queue: queues.Queue,
    *,
    debug_flags: dict[str, Any],
    worker_index: int,
    worker_count: int,
    stats_out_queue: queues.Queue,
) -> Iterator[Any]:
  """Unpickles the element producer from the args queue and closes the queue."""
  (
      serialized_flag_parse_fn,
      serialized_init_fns,
      serialized_element_producer_fn,
  ) = args_queue.get()
  flag_parse_fn: Callable[[Any], None] = cloudpickle.loads(
      serialized_flag_parse_fn
  )
  flag_parse_fn(debug_flags)
  init_fns: list[Callable[[int, int], None]] = cloudpickle.loads(
      serialized_init_fns
  )
  for init_fn in init_fns:
    init_fn(worker_index, worker_count)
  element_producer_fn: GetElementProducerFn[Any] = (
      GetElementProducerFn.deserialize(serialized_element_producer_fn)
  )

  element_producer = element_producer_fn(
      worker_index=worker_index,
      worker_count=worker_count,
      stats_out_queue=stats_out_queue,
  )
  # args_queue has only a single argument and thus can be safely closed.
  args_queue.close()
  return element_producer


def _worker_loop(
    *,
    args_queue: queues.Queue,
    errors_queue: queues.Queue,
    output_queue: queues.Queue,
    termination_event: synchronize.Event,
    worker_index: int,
    worker_count: int,
    enable_profiling: bool,
    debug_flags: dict[str, Any],
    stats_out_queue: queues.Queue,
):
  """Code to be run on each child process."""
  out_of_elements = False
  try:
    worker_index_suffix = "" if worker_count == 1 else f" {worker_index}"
    grain_logging.set_process_identifier_prefix(
        f"PyGrain Worker{worker_index_suffix}"
    )
    logging.info("Starting work.")
    element_producer = _initialize_and_get_element_producer(
        args_queue,
        debug_flags=debug_flags,
        worker_index=worker_index,
        worker_count=worker_count,
        stats_out_queue=stats_out_queue,
    )
    profiling_enabled = enable_profiling and worker_index == 0
    if profiling_enabled:
      profile = cProfile.Profile()
      profile.enable()
    # If termination event is set, we terminate and discard remaining elements.
    while not termination_event.is_set():
      try:
        next_element = next(element_producer)
        if not multiprocessing_common.add_element_to_queue(  # pytype: disable=wrong-arg-types
            next_element, output_queue, termination_event.is_set
        ):
          # We failed to put the element into the output queue because the
          # termination event was set. The element may contain a shared memory
          # block reference that has to be cleaned up.
          _unlink_shm_in_structure(next_element)
      except StopIteration:
        out_of_elements = True
        multiprocessing_common.add_element_to_queue(  # pytype: disable=wrong-arg-types
            _ProcessingComplete(), output_queue, termination_event.is_set
        )
        break
    if profiling_enabled:
      profile.disable()
      _print_profile(f"PROFILE OF PROCESS WITH IDX {worker_index}.", profile)

  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error occurred in child process with worker_index: %i", worker_index
    )
    remote_error = RemoteWorkerError(
        error="".join(
            traceback.format_exception(e.__class__, e, e.__traceback__)
        ),
        worker_index=worker_index,
    )
    try:
      errors_queue.put(remote_error, timeout=_QUEUE_WAIT_TIMEOUT)
    except queue.Full:
      logging.error("Couldn't send exception from child process. Queue full!")

    logging.info(
        "Setting termination event in process with worker_index: %i",
        worker_index,
    )
    termination_event.set()

  if termination_event.is_set():
    if not out_of_elements:
      # Since the termination event is set the consumer will not get any more
      # elements from the output queue. The elements may contain reference to
      # shared memory blocks that have to be cleaned up.
      while not output_queue.empty():
        _unlink_shm_in_structure(output_queue.get_nowait())
    # When adding elements to the queue, element is put in a buffer and a
    # background thread flushes the elements through the pipe. The process that
    # writes to the queue joins that thread automatically on exit. We call
    # cancel_join_thread when system terminates to prevent deadlocks.
    output_queue.cancel_join_thread()
    output_queue.close()
  logging.info("Process %i exiting.", worker_index)


def _unlink_shm_if_metadata(obj: Any):
  if isinstance(obj, shared_memory_array.SharedMemoryArrayMetadata):
    obj.close_and_unlink_shm()


def _unlink_shm_in_structure(structure: Any):
  if isinstance(structure, record.Record):
    _unlink_shm_in_structure(structure.data)
  else:
    tree_lib.map_structure(_unlink_shm_if_metadata, structure)


class GrainPool(Iterator[T]):
  """Pool to parallelize processing of Grain pipelines among a set of processes."""

  def __init__(
      self,
      ctx: context.BaseContext,
      *,
      get_element_producer_fn: GetElementProducerFn[T],
      worker_index_to_start_reading: int = 0,
      termination_event: threading.Event | None = None,
      options: MultiprocessingOptions,
      worker_init_fn: Callable[[int, int], None] | None = None,
      stats_in_queues: tuple[queues.Queue, ...] | None = None,
  ):
    """Initialise a Grain Pool.

    Args:
      ctx: Context to make multiprocessing primitives work.
      get_element_producer_fn: Callable that returns an iterator over the
        elements given the process index and process count.
      worker_index_to_start_reading: index of worker to start reading output
        batches from (needed for checkpointing support).
      termination_event: Setting this event will terminate the pool. Otherwise,
        the pool will terminate when either one of the workers failed or when
        all workers are done processing data. GrainPool will not set this event.
      options: Options for multiprocessing. See MultiprocessingOptions.
      worker_init_fn: Function to run in each worker process before the element
        producer. The function takes two arguments: the current worker index and
        the total worker count.
      stats_in_queues: Queue to propagate execution summary from child processes
        to the parent.
    """
    self.num_processes = options.num_workers
    logging.info("Grain pool will use %i processes.", self.num_processes)
    self.worker_args_queues = []
    self.worker_output_queues = []
    self.processes = []
    # Reader termination should always result in worker termination. However,
    # worker termination should not shut down the reader: workers are terminated
    # when they finished processing data, but the reader may still need to read
    # the remaining output from the shared queues. That is why we use two
    # separate events.
    self._reader_termination_event = termination_event or threading.Event()
    self._workers_termination_event = ctx.Event()
    self._worker_init_fn = worker_init_fn
    self.completed_processes = set()
    # Queue to propagate errors from child processes to the parent. Note that
    # this queue is shared by all child processes.
    self.worker_error_queue = ctx.Queue(self.num_processes)
    self.stats_in_queues = stats_in_queues

    try:
      get_element_producer_fn = get_element_producer_fn.serialize()
    except Exception as e:
      if sys.version_info >= (3, 11):
        e.add_note(
            "\nCould not serialize transformation function passed to Grain "
            "workers. This likely means that your data source, sampler or one "
            "of your transformations cannot be serialized. Please make sure "
            "that the objects work with cloudpickle.dumps()."
        )
      raise e

    for worker_index in range(self.num_processes):
      worker_args_queue = ctx.Queue(1)
      worker_output_queue = ctx.Queue(options.per_worker_buffer_size)
      process_kwargs = dict(
          args_queue=worker_args_queue,
          errors_queue=self.worker_error_queue,
          output_queue=worker_output_queue,
          stats_out_queue=(
              self.stats_in_queues[worker_index]
              if self.stats_in_queues
              else None
          ),
          termination_event=self._workers_termination_event,
          worker_index=worker_index,
          worker_count=options.num_workers,
          enable_profiling=options.enable_profiling,
          debug_flags=dict(
              grain_py_debug_mode=config.get_or_default("py_debug_mode"),
              grain_py_dataset_visualization_output_dir=(
                  config.get_or_default("py_dataset_visualization_output_dir")
              ),
          ),
      )
      # The process kwargs must all be pickable and will be unpickle before
      # absl.app.run() is called. We send arguments via a queue to ensure that
      # they are unpickled after absl.app.run() was called in the child
      # processes.
      worker_init_fns = [self._worker_init_fn] if self._worker_init_fn else []
      parse_debug_flags_fn = parse_debug_flags
      worker_init_fns = cloudpickle.dumps(worker_init_fns)
      parse_debug_flags_fn = cloudpickle.dumps(parse_debug_flags_fn)
      worker_args_queue.put(
          (parse_debug_flags_fn, worker_init_fns, get_element_producer_fn)
      )
      process = ctx.Process(  # pytype: disable=attribute-error  # re-none
          target=_worker_loop, kwargs=process_kwargs, daemon=True
      )
      self.worker_args_queues.append(worker_args_queue)
      self.worker_output_queues.append(worker_output_queue)
      self.processes.append(process)

    logging.info("Grain pool will start child processes.")
    parallel.run_in_parallel(
        function=lambda child_process: child_process.start(),
        list_of_kwargs_to_function=[
            {"child_process": p} for p in self.processes
        ],
        num_workers=min(_PROCESS_MANAGEMENT_MAX_THREADS, self.num_processes),
    )
    logging.info("Grain pool started all child processes.")
    self._next_worker_index = worker_index_to_start_reading

  def __iter__(self) -> GrainPool:
    return self

  def _process_failed(self, worker_index: int) -> bool:
    exit_code = self.processes[worker_index].exitcode
    return exit_code is not None and exit_code != 0

  def _processing_completed(self) -> bool:
    return all(p.exitcode == 0 for p in self.processes)

  def _update_next_worker_index(self) -> None:
    self._next_worker_index = (self._next_worker_index + 1) % self.num_processes

  def __next__(self) -> GrainPoolElement:
    processing_failed = False
    while (
        not self._workers_termination_event.is_set()
        and len(self.completed_processes) < self.num_processes
    ):
      # If the reader was shut down, e.g. due to iterator deletion, we should
      # shut down the workers.
      if self._reader_termination_event.is_set():
        self._shutdown()
        # Since the reader is shut down it doesn't matter what we return here.
        # We should not raise an exception because it is common to iterate over
        # infinite datasets and delete the iterator before processing is
        # complete.
        return GrainPoolElement(
            "Grain worker pool reader was terminated, shutting down workers.",
            -1,
        )
      if self._next_worker_index in self.completed_processes:
        self._update_next_worker_index()
        continue
      try:
        element_worker_index = self._next_worker_index
        element = self.worker_output_queues[self._next_worker_index].get(
            timeout=_QUEUE_WAIT_TIMEOUT
        )
        logging.debug("Read element from process: %s", self._next_worker_index)
        if element == _PROCESSING_COMPLETE:
          logging.info(
              "Processing complete for process with worker_index %i",
              self._next_worker_index,
          )
          self.completed_processes.add(self._next_worker_index)
          self._update_next_worker_index()
        else:
          self._update_next_worker_index()
          return GrainPoolElement(element, element_worker_index)
      except queue.Empty:
        logging.debug("Got no element from process %s", self._next_worker_index)
        if self._process_failed(self._next_worker_index):
          processing_failed = True
          logging.info(
              "Process with idx %i Failed (Exitcode: %s).",
              self._next_worker_index,
              self.processes[self._next_worker_index].exitcode,
          )
          break

    if processing_failed or self._workers_termination_event.is_set():
      logging.error("Processing Failed. Shutting down.")
      self._shutdown()

      try:
        raise self.worker_error_queue.get(timeout=_QUEUE_WAIT_TIMEOUT)
      except queue.Empty:
        # Worker did not report any error. This means that either an exception
        # was raised outside of the worker loop (e.g. during flag parsing) or
        # the worker process was forcefully terminated. Unfortunately, there is
        # no debugging info available in the main process at this point apart
        # from the exit code. The crash logs, however, should've been produced.
        raise RuntimeError(
            f"Grain worker process {self._next_worker_index} was terminated"
            " unexpectedly with exit code "
            f"{self.processes[self._next_worker_index].exitcode}. Search the "
            "logs above for the source of the crash."
        ) from None

    # Processing successfully completed.
    raise StopIteration

  def __del__(self):
    self._shutdown()

  def __enter__(self) -> GrainPool:
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    logging.info("Grain pool is exiting.")
    self._shutdown()

  def _shutdown(self) -> None:
    """Gracefully shutdown the multiprocessing system."""
    logging.info("Shutting down multiprocessing system.")
    try:
      self._workers_termination_event.set()
      # There is a chance that shutdown was triggered before the worker
      # processes fully initialized and read from the arg queues. The arg
      # queues will block the main process until their elements are flushed
      # through the pipes, which will never happen since the workers were shut
      # down. Here we avoid blocking the main process, see
      # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.cancel_join_thread
      for q in self.worker_args_queues:
        q.cancel_join_thread()
        q.close()
      # Not joining here will cause the children to be zombie after they finish.
      # Need to join or call active_children.
      for process in self.processes:
        process.join(timeout=_PROCESS_JOIN_TIMEOUT)
    finally:
      for process in self.processes:
        # In case all our attempts to terminate the system fails, we forcefully
        # kill the child processes.
        if process.is_alive():
          logging.info("Killing worker process with pid %i", process.pid)
          process.kill()


@dataclasses.dataclass(slots=True, frozen=True)
class _ReaderQueueElement:
  """Element to be added to the reader queue."""

  async_result: pool.AsyncResult[Any]
  # index of worker producing the element in [0, worker_count]
  worker_index: int


@dataclasses.dataclass(frozen=True)
class _GrainPoolProcessingComplete:
  """Indicates processing of grain pool is complete."""


_GRAIN_POOL_PROCESSING_COMPLETE = _GrainPoolProcessingComplete()
_QueueElement = Union[
    _ReaderQueueElement, _GrainPoolProcessingComplete, Exception
]


def _open_shared_memory_for_leaf(element: Any) -> Any:
  if isinstance(element, shared_memory_array.SharedMemoryArrayMetadata):
    element = shared_memory_array.SharedMemoryArray.from_metadata(element)
    element.unlink_on_del()
  return element


def _open_shared_memory_for_structure(structure: Any) -> Any:
  if isinstance(structure, record.Record):
    structure.data = tree_lib.map_structure(
        _open_shared_memory_for_leaf, structure.data
    )
    return structure
  return tree_lib.map_structure(_open_shared_memory_for_leaf, structure)


def _process_elements_in_grain_pool(
    *,
    get_element_producer_fn: GetElementProducerFn,
    multiprocessing_options: MultiprocessingOptions,
    reader_queue: queue.Queue[_QueueElement],
    thread_pool: pool.ThreadPool,
    termination_event: threading.Event,
    worker_index_to_start_reading: int,
    worker_init_fn: Callable[[int, int], None] | None,
    stats_in_queues: tuple[queues.Queue, ...] | None,
) -> None:
  """Processes elements in grain worker pool asynchronously."""

  def read_thread_should_stop():
    return termination_event.is_set() or not threading.main_thread().is_alive()

  ctx = mp.get_context("spawn")

  try:
    with GrainPool(
        ctx=ctx,
        get_element_producer_fn=get_element_producer_fn,
        worker_index_to_start_reading=worker_index_to_start_reading,
        termination_event=termination_event,
        options=multiprocessing_options,
        worker_init_fn=worker_init_fn,
        stats_in_queues=stats_in_queues,
    ) as g_pool:
      for element in g_pool:
        if read_thread_should_stop():
          break
        # Note: We use a thread pool for opening the shared memory because
        # in some cases the calls to `shm_open` can actually become the
        # bottleneck for a single thread.
        async_result = thread_pool.apply_async(
            _open_shared_memory_for_structure,
            args=(element.record,),
        )
        multiprocessing_common.add_element_to_queue(
            _ReaderQueueElement(
                async_result,
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


class MultiProcessIteratorInvalidStateError(Exception):
  """Raised when iterator is an invalid state and can't be iterated on."""


class MultiProcessIterator(Iterator[T]):
  """Runs iterators returned by `get_element_producer_fn` in child processes.

  Note: MultiProcessIterator implements the Context Manager protocol to clean
  resources. As such, it must be used within a "with" statement.

  Wraps `GrainPool` adding lifecycle management, multithreaded elements read and
  recording the last worker index useful for checkpointing.
  """

  def __init__(
      self,
      get_element_producer_fn: GetElementProducerFn,
      multiprocessing_options: MultiprocessingOptions,
      worker_index_to_start_reading: int,
      worker_init_fn: Callable[[int, int], None] | None = None,
      stats_in_queues: tuple[queues.Queue, ...] | None = None,
  ):
    """Initializes MultiProcessIterator.

    Args:
      get_element_producer_fn: factory making record iterators for each child
        process.
      multiprocessing_options: options for distributing the record iterators.
      worker_index_to_start_reading: Index of the next worker to read from. This
        is useful for recovering from a checkpoint.
      worker_init_fn: Function to run in each worker process before the element
        producer. The function takes two arguments: the current worker index and
        the total worker count.
      stats_in_queues: Queues to send execution summaries from worker processes
        to the main process.
    """
    self._get_element_producer_fn = get_element_producer_fn
    self._multiprocessing_options = multiprocessing_options
    self._last_worker_index = worker_index_to_start_reading - 1
    self._worker_init_fn = worker_init_fn
    self._reader_queue = None
    self._reader_thread_pool = None
    self._termination_event = None
    self._reader_thread = None
    self._stats_in_queues = stats_in_queues

  def __del__(self):
    if self._reader_thread:
      logging.info("Destroying multiprocess iterator.")
      self.stop_prefetch()

  def start_prefetch(self) -> None:
    """Starts the prefetching threads."""

    if self._reader_thread:
      return

    max_buffered_elements = (
        self._multiprocessing_options.num_workers
        * self._multiprocessing_options.per_worker_buffer_size
    )
    self._reader_queue = queue.Queue(maxsize=max_buffered_elements)
    self._reader_thread_pool = pool.ThreadPool(max_buffered_elements)
    self._termination_event = threading.Event()
    self._reader_thread = threading.Thread(
        target=_process_elements_in_grain_pool,
        kwargs=dict(
            get_element_producer_fn=self._get_element_producer_fn,
            multiprocessing_options=self._multiprocessing_options,
            reader_queue=self._reader_queue,
            thread_pool=self._reader_thread_pool,
            termination_event=self._termination_event,
            worker_index_to_start_reading=self._last_worker_index + 1,
            worker_init_fn=self._worker_init_fn,
            stats_in_queues=self._stats_in_queues,
        ),
    )
    self._reader_thread.start()
    shared_memory_array.SharedMemoryArray.enable_async_del(
        self._multiprocessing_options.num_workers
    )

  def stop_prefetch(self) -> None:
    """Cleans up prefetching threads."""

    if not self._reader_thread:
      return

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

  def __enter__(self):
    self.start_prefetch()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.stop_prefetch()

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
      raise MultiProcessIteratorInvalidStateError(
          "MultiProcessIterator is in an invalid state. Note that"
          " MultiProcessIterator should be used with a 'with' statement."
      )
    return self

  def get_last_worker_index(self):
    return self._last_worker_index

  def __next__(self):
    if not self._can_iterate():
      raise MultiProcessIteratorInvalidStateError(
          "MultiProcessIterator is in an invalid state. Note that"
          " MultiProcessIterator should be used with a 'with' statement."
      )
    element = multiprocessing_common.get_element_from_queue(
        self._reader_queue, self._termination_event.is_set  # pytype: disable=attribute-error
    )
    if isinstance(element, Exception):
      raise element
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
    self._last_worker_index = element.worker_index
    return result
