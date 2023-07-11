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
"""This module provides a way to distribute processing across multiple processes.

GrainPool (not to be confused with Python multiprocessing Pools) allows the
distributing of processing among a set of child processes. The GrainPool works
as follows:
* Parent process launches a set of "num_processes" child processes.
* Data from input iterable is split in a static way among child processes.
* Each child process processes its part of the data and emit resulting elements
  to a queue (each child process has its queue.)
* Parent reads data from the children in a strict round-robin fashion.

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

import cProfile
import dataclasses
from multiprocessing import context
from multiprocessing import queues
from multiprocessing import synchronize
import os
import pstats
import queue
import threading
import traceback
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, Sequence, TypeVar

from absl import logging
import cloudpickle
from grain._src.core import parallel
from grain._src.python import multiprocessing_common

_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")

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


@dataclasses.dataclass
class _SamplerExhausted:
  """Used by feeder thread to indicate that the sampler is exhausted."""


_SAMPLER_EXHAUSTED = _SamplerExhausted()


@dataclasses.dataclass(frozen=True, slots=True)
class GrainPoolElement:
  """Wrapper for output records emited by Grain Pool."""

  record: Any
  worker_idx: Any


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


def _create_input_iterator(
    input_queue: queues.Queue, termination_event: synchronize.Event
) -> Iterator[Any]:
  """Creates iterator that consumes elements from input_queue."""
  while True:
    element = multiprocessing_common.get_element_from_queue(
        input_queue, termination_event.is_set
    )
    # If termination event is set, we receive SystemTerminated.
    if element == multiprocessing_common.SYSTEM_TERMINATED:
      logging.info(
          "Got a SystemTerminated, indicating no more input elements. "
      )
      break
    if element == _SAMPLER_EXHAUSTED:
      logging.info("Sampler exhausted! No more input elements.")
      break
    yield element
  logging.info("Done consuming input iterator.")


def _print_profile(preamble: str, profile: cProfile.Profile):
  """Prints output of cProfile, sorted by cumulative time."""
  print(preamble)
  stats = pstats.Stats(profile).sort_stats(pstats.SortKey.CUMULATIVE)
  stats.print_stats()


def _worker_loop(
    *,
    input_queue: queues.Queue,
    output_queue: queues.Queue,
    errors_queue: queues.Queue,
    args_queue: queues.Queue,
    termination_event: synchronize.Event,
    process_idx: int,
    enable_profiling: bool,
):
  """Code to be run on each child process."""
  try:
    logging.info(
        "Starting work for child process with process_idx: %i", process_idx
    )
    transformation_function = args_queue.get()
    transformation_function = cloudpickle.loads(transformation_function)
    # args_queue has only a single argument and thus can be safely closed.
    args_queue.close()
    profiling_enabled = enable_profiling and process_idx == 0
    if profiling_enabled:
      profile = cProfile.Profile()
      profile.enable()
    iterator_for_process = _create_input_iterator(
        input_queue, termination_event
    )
    iterator_after_transformation = transformation_function(
        iterator_for_process
    )
    # If termination event is set, we terminate and discard remaining elements.
    while not termination_event.is_set():
      try:
        next_element = next(iterator_after_transformation)
        multiprocessing_common.add_element_to_queue(
            next_element, output_queue, termination_event.is_set
        )
      except StopIteration:
        multiprocessing_common.add_element_to_queue(
            _ProcessingComplete(), output_queue, termination_event.is_set
        )
        break
    if profiling_enabled:
      profile.disable()
      _print_profile(f"PROFILE OF PROCESS WITH IDX {process_idx}.", profile)

  except Exception as e:  # pylint: disable=broad-except
    logging.exception(
        "Error occurred in child process with process_idx: %i", process_idx
    )
    remote_error = ExceptionWithTraceback(e, e.__traceback__)
    try:
      errors_queue.put(remote_error, timeout=_QUEUE_WAIT_TIMEOUT)
    except queue.Full:
      logging.error("Couldn't send exception from child process. Queue full!")

    logging.info(
        "Setting termination event in process with process_idx: %i", process_idx
    )
    termination_event.set()

  if termination_event.is_set():
    # When adding elements to the queue, element is put in a buffer and a
    # background thread flushes the elements through the pipe. The process that
    # writes to the queue joins that thread automatically on exit. We call
    # cancel_join_thread when system terminates to prevent deadlocks.
    output_queue.cancel_join_thread()
    output_queue.close()
  logging.info("Process %i exiting.", process_idx)


class GrainPool(Generic[_IN, _OUT]):
  """Pool to parallelize processing of Grain pipelines among a set of processes."""

  def __init__(
      self,
      ctx: context.BaseContext,
      *,
      elements_to_process: Iterable[_IN],
      transformation_function: Callable[[Iterator[_IN]], Iterator[_OUT]],
      num_processes: Optional[int] = None,
      elements_to_buffer_per_process: int = 1,
      enable_profiling: bool = False,
      discard_element_function: Optional[Callable[[_IN, int], bool]] = None,
      worker_idx_to_start_processing: int = 0,
      worker_idx_to_start_reading: int = 0,
  ):
    """Initialise a Grain Pool.

    Args:
      ctx: Context to make multiprocessing primitives work.
      elements_to_process: Iterable of elements that pool should process.
      transformation_function: Function to apply to input elements.
      num_processes: Number of child processes that the pool uses.
      elements_to_buffer_per_process: Number of output elements to buffer per
        process.
      enable_profiling: If True, process with process_idx 0 will be profiled.
      discard_element_function: determine whether an element from the
        elements_to_process should be processed or discarded by the pool.
        Function takes in an element and worker_idx to process the element on.
      worker_idx_to_start_processing: idx of worker that we start distributing
        elements to (needed for checkpointing support).
      worker_idx_to_start_reading: index of worker to start reading output
        batches from (needed for checkpointing support).
    """
    if num_processes is None:
      self.num_processes = os.cpu_count()
      if self.num_processes is None:
        raise NotImplementedError("Cannot determine the number of CPUs.")
    else:
      self.num_processes = num_processes
    logging.info("Grain pool will use %i processes.", self.num_processes)
    logging.info(
        "Grain pool will buffer %i elements per process.",
        elements_to_buffer_per_process,
    )
    logging.info(
        "Grain Pool has profiling enabled."
        if enable_profiling
        else "Grain Pool has profiling disabled."
    )
    self.worker_input_queues = []
    self.worker_output_queues = []
    self.worker_args_queues = []
    self.processes = []
    self.termination_event = ctx.Event()
    self.completed_processes = set()
    # Queue to propagate errors from child processes to the parent. Note that
    # this queue is shared by all child processes.
    self.worker_error_queue = ctx.Queue(self.num_processes)

    try:
      transformation_function = cloudpickle.dumps(transformation_function)
    except Exception as e:
      raise ValueError(
          "Could not serialize transformation_function passed to GrainPool."
          " This likely means that your DataSource, Sampler or one of your"
          " transformations cannot be serialized. Please make sure that the"
          " objects work with cloudpickle.dumps()."
      ) from e

    for process_idx in range(self.num_processes):
      worker_input_queue = ctx.Queue(_INPUT_QUEUE_MAX_SIZE)
      worker_input_queue.cancel_join_thread()
      worker_output_queue = ctx.Queue(elements_to_buffer_per_process)
      worker_args_queue = ctx.Queue(1)
      process_kwargs = {
          "input_queue": worker_input_queue,
          "output_queue": worker_output_queue,
          "errors_queue": self.worker_error_queue,
          "args_queue": worker_args_queue,
          "termination_event": self.termination_event,
          "process_idx": process_idx,
          "enable_profiling": enable_profiling,
      }
      # The process kwargs must all be pickable and will be unpickle before
      # absl.app.run() is called. We send arguments via a queue to ensure that
      # they are unpickled after absl.app.run() was called in the child
      # processes.
      worker_args_queue.put(transformation_function)
      process = ctx.Process(  # pytype: disable=attribute-error  # re-none
          target=_worker_loop, kwargs=process_kwargs, daemon=True
      )
      self.worker_input_queues.append(worker_input_queue)
      self.worker_output_queues.append(worker_output_queue)
      self.worker_args_queues.append(worker_args_queue)
      self.processes.append(process)

    self._feeder_thread = threading.Thread(
        target=self._feed_worker_input_queues,
        args=(
            elements_to_process,
            self.worker_input_queues,
            discard_element_function,
            worker_idx_to_start_processing,
        ),
        daemon=True,
    )
    self._feeder_thread.start()

    logging.info("Grain pool will start child processes.")
    parallel.run_in_parallel(
        function=lambda child_process: child_process.start(),
        list_of_kwargs_to_function=[
            {"child_process": p} for p in self.processes
        ],
        num_workers=min(_PROCESS_MANAGEMENT_MAX_THREADS, self.num_processes),
    )

    logging.info("Grain pool started all child processes.")
    self._next_process_idx = worker_idx_to_start_reading

  def _feed_worker_input_queues(
      self,
      elements_to_process: Iterable[_IN],
      worker_input_queues: Sequence[queues.Queue],
      discard_element_function=None,
      starting_worker: int = 0,
  ) -> None:
    """Puts elements from iterable into processes input queues."""
    queue_idx = starting_worker
    for element in elements_to_process:
      if self.termination_event.is_set():
        break
      if discard_element_function is not None and discard_element_function(
          element, queue_idx
      ):
        queue_idx = (queue_idx + 1) % len(worker_input_queues)
        continue
      multiprocessing_common.add_element_to_queue(
          element, worker_input_queues[queue_idx], self.termination_event.is_set
      )
      queue_idx = (queue_idx + 1) % len(worker_input_queues)
    for worker_input_queue in worker_input_queues:
      multiprocessing_common.add_element_to_queue(
          _SAMPLER_EXHAUSTED, worker_input_queue, self.termination_event.is_set
      )
    logging.info("Feeder thread completed!")

  def __iter__(self) -> GrainPool:
    return self

  def _process_failed(self, process_idx: int) -> bool:
    exit_code = self.processes[process_idx].exitcode
    return exit_code is not None and exit_code != 0

  def _processing_completed(self) -> bool:
    return all(p.exitcode == 0 for p in self.processes)

  def _update_next_process_idx(self) -> None:
    self._next_process_idx = (self._next_process_idx + 1) % self.num_processes

  def __next__(self) -> GrainPoolElement:
    processing_failed = False
    while (
        not self.termination_event.is_set()
        and len(self.completed_processes) < self.num_processes
    ):
      if self._next_process_idx in self.completed_processes:
        self._update_next_process_idx()
        continue
      try:
        element_worker_idx = self._next_process_idx
        element = self.worker_output_queues[self._next_process_idx].get(
            timeout=_QUEUE_WAIT_TIMEOUT
        )
        logging.debug("Read element from process: %s", self._next_process_idx)
        if element == _PROCESSING_COMPLETE:
          logging.info(
              "Processing complete for process with process_idx %i",
              self._next_process_idx,
          )
          self.completed_processes.add(self._next_process_idx)
          self._update_next_process_idx()
        else:
          self._update_next_process_idx()
          return GrainPoolElement(element, element_worker_idx)
      except queue.Empty:
        logging.debug("Got no element from process %s", self._next_process_idx)
        if self._process_failed(self._next_process_idx):
          processing_failed = True
          logging.debug("Process with idx %i Failed.", self._next_process_idx)
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
      self._feeder_thread.join()
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
      for input_queue in self.worker_input_queues:
        # close indicates current process will put no more elements to queue.
        input_queue.close()
