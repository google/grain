# Copyright 2025 Google LLC
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
"""Implements element prefetching transformations with processes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
from multiprocessing import queues
from multiprocessing import sharedctypes
from multiprocessing import synchronize
from multiprocessing import util
import queue
import time
from typing import Any, TypeVar
import weakref

from absl import flags
import cloudpickle
from grain._src.core import monitoring as grain_monitoring
from grain._src.core.config import config
import multiprocessing as mp
from grain._src.python import grain_logging
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import prefetch
from grain._src.python.ipc import queue as grain_queue
from grain._src.python.ipc import shared_memory_array


T = TypeVar("T")

# Type for the iterator state.
StateT = dict[str, Any]

# Timeout for killing worker processes on iterator close.
_PROCESS_KILL_TIMEOUT_S = 25
# Interval to wait in the worker process when the parent iterator is exhausted
# to avoid busy-waiting.
_PARENT_EXHAUSTED_WAIT_S = 0.5
# Timeout for getting an element from the worker process.
_QUEUE_WAIT_TIMEOUT_S = 1

_is_in_worker_process = False


def _run_all(fns: Sequence[Callable[[], None]]):
  for fn in fns:
    fn()


def _parse_debug_flags(debug_flags: dict[str, Any]):
  """Parses debug flags."""
  flags.FLAGS["grain_py_debug_mode"].present = True
  flags.FLAGS["grain_py_dataset_visualization_output_dir"].present = True
  config.update("py_debug_mode", debug_flags["grain_py_debug_mode"])
  config.update(
      "py_dataset_visualization_output_dir",
      debug_flags["grain_py_dataset_visualization_output_dir"],
  )


def _get_dataset_options(ds: dataset.IterDataset) -> base.DatasetOptions:
  result = base.DatasetOptions()
  to_visit = [ds]
  while to_visit:
    parent = to_visit.pop()
    if isinstance(parent, dataset.WithOptionsIterDataset):
      result = result.merge(parent.options)
    to_visit.extend(parent.parents)
  return result


def _check_picklable(
    ds: dataset.IterDataset | dataset.MapDataset,
):
  """Detects the first unpickle-able dataset in post-order.

  Args:
    ds: IterDataset or MapDataset to check whether it is picklable.

  NOTE: This function's time complexity is O(n^2) where n is the number of
  Grain dataset operations because `cloudpickle.dumps(ds)` will trigger
  pickling into all the datasets. If this naive O(n^2) algorithm takes too
  much time, we could consider doing copying `ds`, delete its parents and then
  do `cloudpickle.dumps(new_ds)` to reduce the time complexity to O(n).
  """

  # Traverses the graph in post-order to find the first unpickle-able subtree
  for parent in ds.parents:
    _check_picklable(parent)

  try:
    cloudpickle.dumps(ds)
  except Exception as e:  # pylint: disable=broad-exception-caught
    e.add_note(f"Dataset: {ds} cannot be pickled!")
    raise e


def _serialize_dataset(ds: dataset.IterDataset) -> bytes:
  """Overrides the default implementation to generate better error messages."""
  try:
    return cloudpickle.dumps(ds)
  except Exception as e:  # pylint: disable=broad-except
    # Calls `_check_picklable` to generate useful pickle errors
    _check_picklable(ds)
    # If somehow we cannot find the dataset that is causing the pickle
    # issues, just raise the original error
    raise e


def _clear_queue_and_maybe_unlink_shm(q: queues.Queue[Any]) -> int:
  count = 0
  while True:
    try:
      shared_memory_array.unlink_shm(q.get_nowait())
      count += 1
    except queue.Empty:
      return count


class ProcessPrefetchIterDataset(dataset.IterDataset[T]):
  """Iterable dataset that uses a background process for prefetching.

  This dataset transformation accepts an IterDataset and prefetches elements
  from it in a separate process, buffering up to `buffer_size` elements.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      buffer_size: int,
      worker_init_fn: Callable[[], None] | None = None,
  ):
    """Initializes the ProcessPrefetchIterDataset.

    Args:
      parent: The dataset to prefetch from.
      buffer_size: The size of the buffer used for prefetching.
      worker_init_fn: An optional function to run in the worker process at
        startup.
    """
    if buffer_size <= 0:
      raise ValueError(
          f"`buffer_size` must be greater than 0, got {buffer_size}."
      )
    super().__init__(parent)
    self._buffer_size = buffer_size
    self._worker_init_fn = worker_init_fn

  def __str__(self) -> str:
    return f"ProcessPrefetchIterDataset(buffer_size={self._buffer_size})"

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _ProcessPrefetchDatasetIterator(
        self._parent,
        self._buffer_size,
        self._worker_init_fn,
    )

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


def _put_dataset_elements_in_buffer(
    pickled_parse_debug_flags_fn: bytes,
    pickled_worker_init_fn: bytes,
    pickled_ds: bytes,
    buffer: queues.Queue[
        tuple[Any, StateT | None, int | None, Exception | None]
    ],
    should_stop: synchronize.Event,
    set_state_request_count: sharedctypes.Synchronized[int],
    set_state_queue: queues.Queue[StateT | int],
    stats_out_queue: queues.Queue[Any] | None,
    start_profiling_event: synchronize.Event | None,
    stop_profiling_event: synchronize.Event | None,
    debug_flags: dict[str, Any],
):
  """Prefetches elements in a separate process."""
  global _is_in_worker_process
  _is_in_worker_process = True
  try:
    parse_debug_flags_fn = cloudpickle.loads(pickled_parse_debug_flags_fn)
    parse_debug_flags_fn(debug_flags)
    worker_init_fn = cloudpickle.loads(pickled_worker_init_fn)
    if worker_init_fn is not None:
      worker_init_fn()
    ds = cloudpickle.loads(pickled_ds)
    it = ds.__iter__()
    min_shm_size = it._ctx.dataset_options.min_shm_size  # pylint: disable=protected-access
    # Set the stats queue in worker process to send stats to the main process.
    it._stats._config.stats_out_queue = stats_out_queue  # pylint: disable=protected-access
    parent_exhausted = False
    next_index: int | None = 0
    while not should_stop.is_set():
      if set_state_request_count.value > 0:
        new_state_or_index = None
        with set_state_request_count.get_lock():
          if set_state_request_count.value > 0:
            set_state_request_count.value -= 1
            new_state_or_index = set_state_queue.get()
            parent_exhausted = False
        if new_state_or_index is not None:
          if not grain_queue.add_element_to_queue(  # pytype: disable=wrong-arg-types
              (_SetStateIsDone(), None, None, None),
              buffer,
              should_stop.is_set,
          ):
            continue
          if isinstance(new_state_or_index, int):
            dataset.set_next_index(it, new_state_or_index)
            next_index = new_state_or_index
          else:
            it.set_state(new_state_or_index)
            next_index = None
      if parent_exhausted:
        # Avoid busy-waiting when parent iterator is exhausted due to an
        # error. Wait until set_state_event or should_stop is set.
        time.sleep(_PARENT_EXHAUSTED_WAIT_S)
        continue
      try:
        element = it.__next__()
      except Exception as e:  # pylint: disable=broad-except
        grain_queue.add_element_to_queue(  # pytype: disable=wrong-arg-types
            (None, None, None, e), buffer, should_stop.is_set
        )
        parent_exhausted = True
        continue
      element = shared_memory_array.copy_to_shm(element, min_size=min_shm_size)
      # If the node is prefetch, we already record the bytes produced in it's
      # __next__ method.
      if not it._stats._config.is_prefetch:  # pylint: disable=protected-access
        it._stats.record_bytes_produced(element)  # pylint: disable=protected-access
      if next_index is not None:
        next_index += 1
      if not grain_queue.add_element_to_queue(  # pytype: disable=wrong-arg-types
          (element, it.get_state(), next_index, None),
          buffer,
          should_stop.is_set,
      ):
        # We failed to put the element into the output queue because the
        # should_stop event was set. The element may contain a shared memory
        # block reference that has to be cleaned up.
        shared_memory_array.unlink_shm(element)
  except Exception as e:  # pylint: disable=broad-except
    _clear_queue_and_maybe_unlink_shm(buffer)
    _clear_queue_and_maybe_unlink_shm(set_state_queue)
    grain_queue.add_element_to_queue(  # pytype: disable=wrong-arg-types
        (None, None, None, e), buffer, should_stop.is_set
    )
    return
  _clear_queue_and_maybe_unlink_shm(buffer)
  _clear_queue_and_maybe_unlink_shm(set_state_queue)


class _SetStateIsDone:
  """Placeholder to indicate set_state has completed in worker process."""


def _close_if_alive(
    iterator_ref: weakref.ReferenceType[dataset.DatasetIterator],
):
  """Closes the iterator via weak reference if it's still alive.

  This function is used as a finalizer callback to ensure the iterator is
  closed and resources are released before the multiprocessing module attempts
  to terminate the worker process.

  Args:
    iterator_ref: A weak reference to a DatasetIterator instance.
  """
  iterator = iterator_ref()
  if iterator is not None:
    iterator.close()


class _ProcessPrefetchDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that performs prefetching using a background process."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      buffer_size: int,
      worker_init_fn: Callable[[], None] | None = None,
  ):
    super().__init__()
    self._iter_parent = parent
    self._buffer_size = buffer_size
    self._worker_init_fn = worker_init_fn
    self._keep_workers_after_stop_iteration = False
    # Since the parent iterator is going to be created in each subprocess, and
    # the options are propagated during iterator creation, we need to manually
    # propagate them.
    self._ctx.dataset_options = _get_dataset_options(parent)

    self._process_ctx = mp.get_context("spawn")
    self._state: StateT | None = None
    self._prefetch_process: Any | None = None
    self._prefetch_should_stop: synchronize.Event = self._process_ctx.Event()
    self._set_state_request_count: sharedctypes.Synchronized[int] = (
        self._process_ctx.Value("i", 0)
    )
    self._set_state_queue: queues.Queue[StateT | int] = self._process_ctx.Queue(
        5
    )
    self._buffer: queues.Queue[
        tuple[T, StateT | None, int | None, Exception | None]
    ] = self._process_ctx.Queue(maxsize=self._buffer_size)
    self._stats_in_queue = self._process_ctx.Queue(maxsize=5)
    self._start_profiling_event = self._process_ctx.Event()
    self._stop_profiling_event = self._process_ctx.Event()
    self._set_state_count = 0
    self._exhausted = False
    self._prefetch_ds_iter = None
    self._next_index: int | None = 0
    # Ensure that the iterator is closed before the multiprocessing module
    # attempts to terminate the worker processes to prevent process hanging
    # issues.
    util.Finalize(
        self,
        _close_if_alive,
        args=(weakref.ref(self),),
        exitpriority=1,
    )

  # pytype: disable=attribute-error
  # pylint: disable=protected-access
  def _initialize_stats(
      self, execution_tracking_mode: base.ExecutionTrackingMode
  ):
    # This method is needed to set `is_prefetch` to `True` in the stats config.
    stats_config = dataset_stats.StatsConfig(
        name=str(self),
        transform_mutates_spec=self._MUTATES_ELEMENT_SPEC,
        node_type=dataset_stats.NodeType.PREFETCH,
        iter_weakref=dataset_stats.HashableWeakRef(self),
    )
    if self._stats_in_queue is not None:
      stats_config.stats_in_queues = (self._stats_in_queue,)

    # If the stats object has already been initialized, copy the queues from
    # the original stats object to the new stats object.
    if "_stats" in self.__dict__:
      stats_config.stats_out_queue = self._stats._config.stats_out_queue
      stats_config.stats_in_queues = self._stats._config.stats_in_queues

    return dataset_stats.make_stats(
        stats_config,
        [],
        execution_tracking_mode=execution_tracking_mode,
    )

  @functools.cached_property
  def _stats(self):
    return self._initialize_stats(
        self._ctx.dataset_options.execution_tracking_mode
    )

  # pytype: enable=attribute-error
  # pylint: enable=protected-access

  def start_prefetch(self) -> None:
    """Starts prefetching elements in background.

    Raises:
      ValueError: If the iterator has been closed.
    """
    if self._closed:
      raise ValueError("Attempting to use a closed iterator.")
    if self._prefetch_process is not None:
      return

    self._prefetch_should_stop.clear()
    ds = dataset.WithOptionsIterDataset(
        self._iter_parent,
        options=self._ctx.dataset_options,
    )
    self._prefetch_process = self._process_ctx.Process(
        target=_put_dataset_elements_in_buffer,
        kwargs=dict(
            pickled_parse_debug_flags_fn=cloudpickle.dumps(_parse_debug_flags),
            pickled_worker_init_fn=cloudpickle.dumps(self._worker_init_fn),
            pickled_ds=_serialize_dataset(ds),
            buffer=self._buffer,
            should_stop=self._prefetch_should_stop,
            set_state_request_count=self._set_state_request_count,
            set_state_queue=self._set_state_queue,
            stats_out_queue=self._stats_in_queue,
            start_profiling_event=self._start_profiling_event,
            stop_profiling_event=self._stop_profiling_event,
            debug_flags=dict(
                grain_py_debug_mode=config.get_or_default("py_debug_mode"),
                grain_py_dataset_visualization_output_dir=(
                    config.get_or_default("py_dataset_visualization_output_dir")
                ),
            ),
        ),
        daemon=True,
        name=f"grain-process-prefetch-{str(self)}",
    )
    self._prefetch_process.start()
    shared_memory_array.SharedMemoryArray.enable_async_del(1)

  def _process_failed(self) -> bool:
    if self._prefetch_process is None:
      return False
    exit_code = self._prefetch_process.exitcode
    return exit_code is not None and exit_code != 0

  @dataset_stats.record_next_duration_if_output
  def __next__(self):
    if self._exhausted:
      raise StopIteration
    timer = dataset_stats.Timer()
    with timer:
      self.start_prefetch()
      # Loop until we get a non-stale element.
      while True:
        try:
          element, state, next_index, err = self._buffer.get(
              timeout=_QUEUE_WAIT_TIMEOUT_S
          )
        except queue.Empty:
          assert self._prefetch_process is not None
          if self._process_failed():
            element, state, next_index = None, None, None
            err = RuntimeError(
                "Worker process was terminated unexpectedly with exit code "
                f"{self._prefetch_process.exitcode}. Search the logs above for "
                "the source of the crash."
            )
          else:
            continue
        if isinstance(element, _SetStateIsDone):
          self._set_state_count -= 1
        elif self._set_state_count == 0:
          break
        elif element is not None:
          # Unlink shared memory for the discarded element.
          shared_memory_array.unlink_shm(element)
      if err is not None:
        if (
            not isinstance(err, StopIteration)
            or not self._keep_workers_after_stop_iteration
        ):
          self._stop_prefetch()
        self._exhausted = True
        raise err
      self._state = state
      self._next_index = next_index
    with self._stats.record_self_time(offset_ns=timer.value()):
      element = self._stats.record_bytes_produced(element)
      return shared_memory_array.open_from_shm(element)

  def close(self):
    """Stops the iterator. No further calls to the iterator are expected."""
    self._closed = True
    self._stop_prefetch()

  def _clear_set_state_queue(self):
    with self._set_state_request_count.get_lock():
      num_removed = _clear_queue_and_maybe_unlink_shm(self._set_state_queue)
      self._set_state_count -= num_removed
      self._set_state_request_count.value -= num_removed

  def _stop_prefetch(self):
    """Stops the prefetching process if it's currently running."""
    if self._prefetch_process is None:
      return

    self._prefetch_should_stop.set()
    _clear_queue_and_maybe_unlink_shm(self._buffer)
    self._clear_set_state_queue()

    # Not joining here will cause the children to be zombie after they finish.
    # Need to join or call active_children.
    self._prefetch_process.join(timeout=_PROCESS_KILL_TIMEOUT_S)

    # In case all our attempts to terminate the system fails, we forcefully
    # kill the child processes.
    if self._prefetch_process.is_alive():
      self._prefetch_process.kill()
    else:
      _clear_queue_and_maybe_unlink_shm(self._buffer)
    self._prefetch_process = None
    self._set_state_count = 0

  def get_state(self) -> StateT:
    if self._state is not None:
      return self._state
    elif self._next_index == 0:
      self._state = self._iter_parent.__iter__().get_state()
      return self._state
    else:
      # This point is only reached if `get_state` is called after
      # `set_next_index` and before the next `__next__` call. Get the state
      # corresponding to the new `next_index`.
      ds_iter = self._iter_parent.__iter__()
      dataset.set_next_index(ds_iter, self._next_index)
      self._state = ds_iter.get_state()
      return self._state

  def set_state(self, state: StateT):
    self._state = state
    # Remove any pending set_state calls.
    self._clear_set_state_queue()
    self._set_state_queue.put(self._state)
    # Signal the prefetch process to start processing set_state calls.
    with self._set_state_request_count.get_lock():
      self._set_state_request_count.value += 1
    # Increment the number of _SetStateIsDone that need to be skipped to
    # avoid stale elements in the buffer.
    self._set_state_count += 1
    self._exhausted = False
    self._next_index = None

  def _get_next_index(self) -> int:
    if self._next_index is not None:
      return self._next_index
    else:
      # Need to create iterator in main process to get the correct index if
      # `_get_next_index` is called after calling `set_state`.
      ds_iter = self._iter_parent.__iter__()
      ds_iter.set_state(self._state)
      self._next_index = dataset.get_next_index(ds_iter)
      return self._next_index

  def _set_next_index(self, next_index: int):
    # If `_set_next_index` is not supported for a parent iterator, the error
    # will not show up until the next `__next__` call.
    self._next_index = next_index
    # Remove any pending set_state calls.
    self._clear_set_state_queue()
    self._set_state_queue.put(self._next_index)
    # Signal the prefetch process to start processing set_state calls.
    with self._set_state_request_count.get_lock():
      self._set_state_request_count.value += 1
    # Increment the number of _SetStateIsDone that need to be skipped to
    # avoid stale elements in the buffer.
    self._set_state_count += 1
    self._exhausted = False
    self._state = None

  def set_keep_workers_after_stop_iteration(self, keep_workers: bool):
    # Determines whether the worker processes should be kept alive after
    # StopIteration is raised by `__next__`. This is used by
    # `RepeatDatasetIterator` to allow for resetting the iterator state and
    # continuing iteration without recreating the worker processes.
    self._keep_workers_after_stop_iteration = keep_workers

  def __str__(self) -> str:
    return f"ProcessPrefetchDatasetIterator(buffer_size={self._buffer_size})"


class _LazyWorkerSliceIterDataset(dataset.IterDataset[T]):
  """Applies slice to the parent dataset in the worker process."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      sl: slice,
      sequential_slice: bool,
  ):
    super().__init__(parent)
    self._slice = sl
    self._sequential_slice = sequential_slice

  def __iter__(self) -> dataset.DatasetIterator[T]:
    if not _is_in_worker_process:
      return self._parent.__iter__()
    prefetch._set_slice_iter_dataset(
        self._parent, self._slice, self._sequential_slice
    )
    return self._parent.__iter__()

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)

  def __str__(self) -> str:
    return f"_LazyWorkerSliceIterDataset(slice={self._slice})"


def multiprocess_prefetch(
    ds: dataset.IterDataset[T],
    num_workers: int = 0,
    buffer_size: int = 1,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> dataset.IterDataset[T]:
  """Uses a multiple processes to prefetch elements ahead of time.

  It works by sharding the input dataset into `num_workers` shards, and
  interleaving them. Each shard is read by a separate process inside
  `InterleaveIterDataset`.

  Args:
    ds: The parent dataset to prefetch from.
    num_workers: The number of processes to use for prefetching. If 0,
      prefetching is disabled and this is a no-op.
    buffer_size: The size of the prefetch buffer for each process.
    worker_init_fn: A function that is called in each worker process.
    sequential_slice: Whether to use sequential slicing.

  Returns:
    `IterDataset` that prefetches elements from `ds` using multiple processes.
  """
  if num_workers == 0:
    return ds

  dataset_options = _get_dataset_options(ds)

  shards = []
  for i in range(num_workers):
    if num_workers == 1:
      worker_ds = ds
    else:
      worker_ds = _LazyWorkerSliceIterDataset(
          ds,
          slice(i, None, num_workers),
          sequential_slice,
      )

    worker_ds = prefetch._MpContextIterDataset(  # pylint: disable=protected-access
        worker_ds,
        base.MultiprocessingContext(
            process_index=i,
            process_count=num_workers,
        ),
    )
    worker_index_suffix = "" if num_workers == 1 else f" {i}"

    worker_init_fns = [
        functools.partial(
            grain_logging.set_process_identifier_prefix, worker_index_suffix
        )
    ]
    if worker_init_fn is not None:
      worker_init_fns.append(functools.partial(worker_init_fn, i, num_workers))
    worker_ds = ProcessPrefetchIterDataset(
        worker_ds,
        buffer_size=buffer_size,
        worker_init_fn=functools.partial(_run_all, worker_init_fns),
    )
    shards.append(worker_ds)

  ds = interleave.InterleaveIterDataset(
      shards,
      cycle_length=num_workers,
      num_make_iter_threads=num_workers,
      make_iter_buffer_size=num_workers,
      iter_buffer_size=buffer_size,
  )
  # Apply options from parent dataset because interleave dataset does not
  # propagate options.
  ds = dataset.WithOptionsIterDataset(ds, dataset_options)
  return ds
