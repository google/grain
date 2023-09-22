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
each machine runs one Python process (identified by `jax.proccess_index()`).
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
  Parent and child processes continously check if the "termination_event" and if
  set, they break from what they are doing.
* We never block indefinitely when calling get() or put() on a queue. This
  ensures parent and child processes continue to check the termination_event.
"""

from __future__ import annotations

from collections.abc import Iterator
import cProfile
import dataclasses
from multiprocessing import context
from multiprocessing import queues
from multiprocessing import synchronize
import pstats
import queue
import traceback
from typing import Any, Protocol, TypeVar

from absl import logging
import cloudpickle
from grain._src.core import parallel
from grain._src.python import grain_logging
from grain._src.python import multiprocessing_common
from grain._src.python.experimental.shared_memory import np_array_in_shared_memory
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


@dataclasses.dataclass(frozen=True, slots=True)
class GrainPoolElement:
  """Wrapper for output records emited by Grain Pool."""

  record: Any
  worker_index: Any


# Hack to embed stringification of remote traceback in local traceback.
class RemoteTracebackError(Exception):

  def __init__(self, tb: traceback.TracebackType):
    self.tb = tb

  def __str__(self):
    return self.tb


class ExceptionWithTraceback:
  """Exception that can be pickled and sent over the queue."""

  def __init__(self, exception: Exception, tb: traceback.TracebackType):
    tb = traceback.format_exception(type(exception), exception, tb)
    tb = "".join(tb)
    self.exception = exception
    self.tb = '\n"""\n%s"""' % tb

  def __reduce__(self):
    return rebuild_exception, (self.exception, self.tb)


def rebuild_exception(exception: Exception, tb: traceback.TracebackType):
  """Rebuilds the exception at the received side."""
  exception.__cause__ = RemoteTracebackError(tb)
  return exception


def _print_profile(preamble: str, profile: cProfile.Profile):
  """Prints output of cProfile, sorted by cumulative time."""
  print(preamble)
  stats = pstats.Stats(profile).sort_stats(pstats.SortKey.CUMULATIVE)
  stats.print_stats()


class GetElementProducerFn(Protocol[T]):

  def __call__(self, *, worker_index: int, worker_count: int) -> Iterator[T]:
    """Returns a generator of elements."""


def _get_element_producer_from_queue(
    args_queue: queues.Queue, *, worker_index: int, worker_count: int
) -> Iterator[Any]:
  """Unpickles the element producer from the args queue and closes the queue."""
  serialized_element_producer_fn = args_queue.get()
  element_producer_fn: GetElementProducerFn[Any] = cloudpickle.loads(
      serialized_element_producer_fn
  )
  element_producer = element_producer_fn(
      worker_index=worker_index, worker_count=worker_count
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
    enable_numpy_shared_memory: bool = False,
):
  """Code to be run on each child process."""
  try:
    grain_logging.set_process_identifier_prefix(
        f"PyGrain Worker {worker_index}"
    )
    if enable_numpy_shared_memory:
      np_array_in_shared_memory.enable_numpy_shared_memory_pickler()
    logging.info("Starting work.")
    element_producer = _get_element_producer_from_queue(
        args_queue, worker_index=worker_index, worker_count=worker_count
    )
    profiling_enabled = enable_profiling and worker_index == 0
    if profiling_enabled:
      profile = cProfile.Profile()
      profile.enable()
    # If termination event is set, we terminate and discard remaining elements.
    while not termination_event.is_set():
      try:
        next_element = next(element_producer)
        multiprocessing_common.add_element_to_queue(  # pytype: disable=wrong-arg-types
            next_element, output_queue, termination_event.is_set
        )
      except StopIteration:
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
    remote_error = ExceptionWithTraceback(e, e.__traceback__)
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
    # When adding elements to the queue, element is put in a buffer and a
    # background thread flushes the elements through the pipe. The process that
    # writes to the queue joins that thread automatically on exit. We call
    # cancel_join_thread when system terminates to prevent deadlocks.
    output_queue.cancel_join_thread()
    output_queue.close()
  logging.info("Process %i exiting.", worker_index)


class GrainPool(Iterator[T]):
  """Pool to parallelize processing of Grain pipelines among a set of processes."""

  def __init__(
      self,
      ctx: context.BaseContext,
      *,
      get_element_producer_fn: GetElementProducerFn[T],
      worker_index_to_start_reading: int = 0,
      options: MultiprocessingOptions,
  ):
    """Initialise a Grain Pool.

    Args:
      ctx: Context to make multiprocessing primitives work.
      get_element_producer_fn: Callable that returns an iterator over the
        elements given the process index and process count.
      worker_index_to_start_reading: index of worker to start reading output
        batches from (needed for checkpointing support).
      options: Options for multiprocessing. See MultiprocessingOptions.
    """
    self.num_processes = options.num_workers
    logging.info("Grain pool will use %i processes.", self.num_processes)
    self.worker_args_queues = []
    self.worker_output_queues = []
    self.processes = []
    self.termination_event = ctx.Event()
    self.completed_processes = set()
    # Queue to propagate errors from child processes to the parent. Note that
    # this queue is shared by all child processes.
    self.worker_error_queue = ctx.Queue(self.num_processes)

    try:
      get_element_producer_fn = cloudpickle.dumps(get_element_producer_fn)
    except Exception as e:
      raise ValueError(
          "Could not serialize transformation_function passed to GrainPool."
          " This likely means that your DataSource, Sampler or one of your"
          " transformations cannot be serialized. Please make sure that the"
          " objects work with cloudpickle.dumps()."
      ) from e

    for worker_index in range(self.num_processes):
      worker_args_queue = ctx.Queue(1)
      worker_output_queue = ctx.Queue(options.per_worker_buffer_size)
      process_kwargs = {
          "args_queue": worker_args_queue,
          "errors_queue": self.worker_error_queue,
          "output_queue": worker_output_queue,
          "termination_event": self.termination_event,
          "worker_index": worker_index,
          "worker_count": options.num_workers,
          "enable_profiling": options.enable_profiling,
          "enable_numpy_shared_memory": (
              np_array_in_shared_memory.numpy_shared_memory_pickler_enabled()
          ),
      }
      # The process kwargs must all be pickable and will be unpickle before
      # absl.app.run() is called. We send arguments via a queue to ensure that
      # they are unpickled after absl.app.run() was called in the child
      # processes.
      worker_args_queue.put(get_element_producer_fn)
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
        not self.termination_event.is_set()
        and len(self.completed_processes) < self.num_processes
    ):
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
          logging.debug("Process with idx %i Failed.", self._next_worker_index)
          break

    if processing_failed or self.termination_event.is_set():
      logging.error("Processing Failed. Shutting down.")
      self._shutdown()

      exception_to_raise = Exception("Processing Failed. Shutting down.")
      try:
        remote_error = self.worker_error_queue.get(timeout=_QUEUE_WAIT_TIMEOUT)
        logging.error("Got error %s", remote_error)
        exception_to_raise.__cause__ = remote_error
      except queue.Empty:
        logging.error("Can't determine remote exception!")
      raise exception_to_raise

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
      self.termination_event.set()
      # Not joining here will cause the children to be zombie after they finish.
      # Need to join or call active_children.
      for process in self.processes:
        process.join(timeout=_PROCESS_JOIN_TIMEOUT)
    finally:
      for process in self.processes:
        # In case all our attempts to terminate the system fails, we forcefully
        # kill the child processes.
        if process.is_alive():
          logging.info("Forcibly terminating process with pid %i", process.pid)
          process.terminate()
