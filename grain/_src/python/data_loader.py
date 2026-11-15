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
import contextlib
import dataclasses
import functools
import json
import os
import sys
import time
from typing import Any, Awaitable, Callable, Optional, Sequence, TypeVar

from etils import epath
from grain._src.core import monitoring
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import tree_lib
from grain._src.python import operations as ops
from grain._src.python import options
from grain._src.python import record
from grain._src.python import samplers
from grain._src.python.checkpoint import base as checkpoint_base
from grain._src.python.dataset import base as dataset_base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import batch as batch_ds
from grain._src.python.dataset.transformations import flatmap
from grain._src.python.dataset.transformations import source as source_ds
from grain._src.python.ipc import shared_memory_array
import numpy as np


_api_usage_counter = monitoring.Counter(
    "/grain/python/data_loader/api",
    monitoring.Metadata(description="API initialization counter."),
    fields=[("name", str)],
)
_iterator_get_next_metric = monitoring.EventMetric(
    "/grain/python/data_loader/iterator_get_next",
    monitoring.Metadata(
        description="Histogram for DataLoaderIterator.__next__() latency.",
        units=monitoring.Units.NANOSECONDS,
    ),
    bucketer=monitoring.Bucketer.PowersOf(2.0),
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


def _determine_worker_count(input_worker_count: int | None) -> int:
  """Determines count of child processes to use."""
  if input_worker_count is not None:
    return int(input_worker_count)
  if (os_cpu_count := os.cpu_count()) is not None:
    return os_cpu_count
  else:
    raise ValueError("Can't determine worker count. Please set worker count.")


@contextlib.contextmanager
def use_context_if_available(obj):
  """Uses with statement if obj is a context manager, else just uses the object."""
  if hasattr(obj, "__enter__") and hasattr(obj, "__exit__"):
    with obj:
      yield
  else:
    yield


@dataclasses.dataclass
class CopyNumPyArrayToSharedMemory(transforms.Map):
  """If `element` contains NumPy array copy it to SharedMemoryArray."""

  def map(self, element: Any) -> Any:
    def copy_if_applied(element: Any) -> Any:
      if (
          not isinstance(element, np.ndarray)
          or element.dtype.hasobject
          or not element.flags.c_contiguous
      ):
        return element

      shared_memory_arr = shared_memory_array.SharedMemoryArray(
          element.shape, element.dtype
      )
      np.copyto(shared_memory_arr, element, casting="no")
      return shared_memory_arr.metadata

    return tree_lib.map_structure(copy_if_applied, element)


class _SamplerMapDataset(dataset.MapDataset[record.Record]):
  """A MapDataset that is indexed by a sampler."""

  def __init__(
      self,
      data_source: dataset_base.RandomAccessDataSource,
      sampler: samplers.Sampler,
      shard_options: sharding.ShardOptions,
  ):
    super().__init__(dataset.MapDataset.source(data_source))
    self._sampler = sampler
    self._shard_options = shard_options
    sampler_size = self._sampler_size()
    shard_count = self._shard_options.shard_count
    if self._shard_options.drop_remainder:
      self.length = sampler_size // shard_count
    else:
      remainder = sampler_size % shard_count
      self.length = sampler_size // shard_count + (
          1 if self._shard_options.shard_index < remainder else 0
      )

  def _sampler_size(self) -> int:
    """Returns the length of the sampler."""
    if hasattr(self._sampler, "__len__"):
      return len(self._sampler)  # pyrefly: ignore[bad-argument-type]
    lo = 0
    hi = sys.maxsize
    while lo < hi:
      mid = (lo + hi) // 2
      try:
        self._sampler[mid]  # pylint: disable=pointless-statement
        lo = mid + 1
      except IndexError:
        hi = mid
    return lo

  def __len__(self) -> int:
    return self.length

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)

    # Account for sharding
    index = (
        index * self._shard_options.shard_count
        + self._shard_options.shard_index
    )
    with self._stats.record_self_time():
      metadata = self._sampler[index]
      return record.Record(
          metadata=metadata, data=self._parent[metadata.record_key]  # pyrefly: ignore[bad-index]
      )


class _OperationIterDataset(dataset.IterDataset[_T]):
  """Applies an operation to the dataset.

  NOTE: This class should not be used anywhere else as operations break
  checkpointing. It only exists for legacy reasons and backwards compatibility.
  """

  def __init__(
      self,
      parent: dataset.IterDataset[_T],
      operation: ops.Operation,
  ):
    super().__init__(parent)
    self.operation = operation

  def __iter__(self) -> dataset.DatasetIterator[_T]:
    return _OperationDatasetIterator(self._parent.__iter__(), self.operation)


class _OperationDatasetIterator(dataset.DatasetIterator[_T]):
  """Iterator that applies an operation to the dataset."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[_T],
      operation: ops.Operation,
  ):
    super().__init__(parent)
    self._operation = operation
    self._iterator = operation(parent)  # pyrefly: ignore[bad-argument-type]

  def __next__(self) -> _T:
    return next(self._iterator)  # pyrefly: ignore[bad-return]

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)


_STR_ONLY_TYPES = frozenset((str,))


def _is_canonical_data_sources_module(mod: str) -> bool:
  return mod == "grain._src.python.data_sources" or mod.endswith(
      ".grain._src.python.data_sources"
  )


def _is_canonical_array_record_module(mod: str) -> bool:
  return (
      _is_canonical_data_sources_module(mod)
      or mod == "array_record.python.array_record_data_source"
      or mod.endswith(".array_record.python.array_record_data_source")
  )


@dataclasses.dataclass(slots=True)
class _StateReprCache:
  """Caches checkpoint reprs strictly for built-in sources and samplers."""

  _sampler_entry: tuple[Any, str] | None = None
  _data_source_entry: tuple[Any, int, Any, str] | None = None

  def get_sampler_repr(self, sampler: samplers.Sampler) -> str:
    """Returns cached or freshly computed `repr(sampler)`."""
    cls = sampler.__class__
    # Exact class match only (rejects custom samplers and subclasses).
    # pylint: disable=protected-access
    if cls is samplers.IndexSampler:
      s: Any = sampler
      so = s._shard_options
      so_cls = so.__class__
      if so_cls is sharding.ShardOptions or so_cls is sharding.NoSharding:
        key: Any = (
            cls,
            s._num_records,
            so_cls,
            so.shard_index,
            so.shard_count,
            so.drop_remainder,
            s._shuffle,
            s._num_epochs,
            s._seed,
        )
        entry = self._sampler_entry
        if entry is not None and entry[0] == key:
          return entry[1]
        repr_str = repr(sampler)
        self._sampler_entry = (key, repr_str)
        return repr_str
    elif cls is samplers.SequentialSampler:
      s_seq: Any = sampler
      so = s_seq._shard_options
      so_cls = so.__class__
      if so_cls is sharding.ShardOptions or so_cls is sharding.NoSharding:
        key = (
            cls,
            s_seq._num_records,
            so_cls,
            so.shard_index,
            so.shard_count,
            so.drop_remainder,
        )
        entry = self._sampler_entry
        if entry is not None and entry[0] == key:
          return entry[1]
        repr_str = repr(sampler)
        self._sampler_entry = (key, repr_str)
        return repr_str
    # pylint: enable=protected-access
    return repr(sampler)

  def get_data_source_repr(
      self, data_source: dataset_base.RandomAccessDataSource
  ) -> str:
    """Returns cached or freshly computed `_source_repr(data_source)`."""
    cls = data_source.__class__
    entry = self._data_source_entry
    if entry is not None and entry[0] is cls:
      kind = entry[1]
      if kind == 0:
        return entry[3]
      if kind == 1:
        ds_any: Any = data_source
        # pylint: disable=protected-access
        if (ds_any._start, ds_any._stop, ds_any._step) == entry[2]:
          return entry[3]
        # pylint: enable=protected-access
      else:
        paths = getattr(data_source, "_paths", None)
        if paths.__class__ in (list, tuple):
          paths_tuple = tuple(paths)
          if (
              paths_tuple == entry[2]
              and set(map(type, paths_tuple)) <= _STR_ONLY_TYPES
          ):
            return entry[3]

    if cls in (list, tuple, np.ndarray):
      repr_str = str(cls)
      self._data_source_entry = (cls, 0, None, repr_str)
      return repr_str
    mod = cls.__module__
    if cls.__name__ == "RangeDataSource" and _is_canonical_data_sources_module(
        mod
    ):
      key = (
          getattr(data_source, "_start", None),
          getattr(data_source, "_stop", None),
          getattr(data_source, "_step", None),
      )
      repr_str = _source_repr(data_source)
      self._data_source_entry = (cls, 1, key, repr_str)
      return repr_str
    if (
        cls.__name__ == "ArrayRecordDataSource"
        and _is_canonical_array_record_module(mod)
    ):
      paths = getattr(data_source, "_paths", None)
      if paths.__class__ in (list, tuple) and (
          set(map(type, paths)) <= _STR_ONLY_TYPES
      ):
        paths_tuple = tuple(paths)
        repr_str = _source_repr(data_source)
        self._data_source_entry = (cls, 2, paths_tuple, repr_str)
        return repr_str
    return _source_repr(data_source)


class _DataLoaderStateIterDataset(dataset.IterDataset[_T]):
  """IterDataset that converts between dataset and data loader state."""

  def __init__(
      self,
      parent: dataset.IterDataset[_T],
      shard_options: sharding.ShardOptions,
      worker_count: int,
      sampler: samplers.Sampler,
      data_source: dataset_base.RandomAccessDataSource,
      repr_cache: _StateReprCache | None = None,
  ):
    super().__init__(parent)
    self._shard_options = shard_options
    self._worker_count = worker_count
    self._sampler = sampler
    self._data_source = data_source
    self._repr_cache = (
        repr_cache if repr_cache is not None else _StateReprCache()
    )

  def __iter__(self) -> dataset.DatasetIterator[_T]:
    return _DataLoaderStateDatasetIterator(
        self._parent.__iter__(),
        self._shard_options,
        self._worker_count,
        self._sampler,
        self._data_source,
        self._repr_cache,
    )


class _DataLoaderStateDatasetIterator(dataset.DatasetIterator[_T]):
  """Iterator that converts between dataset and data loader state."""

  def __init__(
      self,
      parent: dataset.DatasetIterator[_T],
      shard_options: sharding.ShardOptions | None,
      worker_count: int,
      sampler: samplers.Sampler,
      data_source: dataset_base.RandomAccessDataSource,
      repr_cache: _StateReprCache | None = None,
  ):
    super().__init__(parent)
    self._shard_options = shard_options
    self._worker_count = worker_count
    self._sampler = sampler
    self._data_source = data_source
    self._repr_cache = (
        repr_cache if repr_cache is not None else _StateReprCache()
    )
    self._shard_index = shard_options.shard_index if shard_options else 0
    self._shard_count = shard_options.shard_count if shard_options else 1
    self._global_worker_count = worker_count * self._shard_count
    self._local_offset = self._shard_index - self._global_worker_count
    self._worker_index_keys = tuple(str(i) for i in range(worker_count))

  def __next__(self) -> _T:
    return next(self._parent)

  def get_state(self):
    # TODO: Fix user tests relying on the exact checkpoint
    # structure and switch to native dataset checkpointing. This class can be
    # removed afterwards.
    dataset_state = self._parent.get_state()
    shard_index = self._shard_index
    shard_count = self._shard_count
    if "iterators_in_use_states" not in dataset_state:
      last_worker_index = -1
      last_seen_indices = {
          "0": int(
              shard_index + (dataset_state["next_index"] - 1) * shard_count
          )
      }
    else:
      last_worker_index = dataset_state["next_index_in_cycle"] - 1
      workers_state = dataset_state["iterators_in_use_states"]
      worker_count = len(workers_state)
      if worker_count == 1:
        last_seen_indices = {
            "0": int(
                shard_index + (workers_state[0]["next_index"] - 1) * shard_count
            )
        }
      else:
        if worker_count == self._worker_count:
          global_worker_count = self._global_worker_count
          local_offset = self._local_offset
          keys = self._worker_index_keys
        else:
          global_worker_count = worker_count * shard_count
          local_offset = shard_index - global_worker_count
          keys = tuple(str(i) for i in range(worker_count))
        last_seen_indices = {
            keys[i]: int(
                local_offset
                + i * shard_count
                + workers_state[i]["next_index"] * global_worker_count
            )
            for i in range(worker_count)
        }

    return {
        _VERSION: _CHECKPOINT_VERSION_NUMBER,
        _LAST_SEEN_INDICES: last_seen_indices,
        _LAST_WORKER_INDEX: last_worker_index,
        _WORKER_COUNT: self._worker_count,
        _SAMPLER: self._repr_cache.get_sampler_repr(self._sampler),
        _DATA_SOURCE: self._repr_cache.get_data_source_repr(self._data_source),
    }

  def set_state(self, state):
    last_seen_indices = state[_LAST_SEEN_INDICES]
    last_worker_index = state[_LAST_WORKER_INDEX]
    worker_count = state[_WORKER_COUNT]
    shard_count = self._shard_count
    shard_index = self._shard_index

    if worker_count == 0:
      dataset_state = {
          "next_index": (
              (last_seen_indices["0"] + shard_count - shard_index)
              // shard_count
          )
      }
      self._parent.set_state(dataset_state)
      return

    if worker_count == self._worker_count:
      global_worker_count = self._global_worker_count
      keys = self._worker_index_keys
    else:
      global_worker_count = worker_count * shard_count
      keys = tuple(str(i) for i in range(worker_count))

    iterators_in_use_states = [
        {
            "next_index": (
                (
                    last_seen_indices[keys[worker_index]]
                    + global_worker_count
                    - shard_index
                    - worker_index * shard_count
                )
                // global_worker_count
            )
        }
        for worker_index in range(worker_count)
    ]

    dataset_state = {
        "next_index_in_cycle": (last_worker_index + 1) % worker_count,
        "next_index_in_datasets": worker_count,
        "iterators_in_use_indices": list(range(worker_count)),
        "iterators_in_use_states": iterators_in_use_states,
        "exhausted": [False] * worker_count,
    }
    self._parent.set_state(dataset_state)


class _DataLoaderPipelineMarkerIterDataset(dataset.IterDataset):
  """Marks iterators of this dataset as part of a DataLoader pipeline."""

  def __iter__(self) -> dataset.DatasetIterator:
    iterator = self._parent.__iter__()
    iterator._ctx.is_dataloader_pipeline = True  # pylint: disable=protected-access
    return iterator


class DataLoader:
  """DataLoader loads and transforms input data."""

  def __init__(
      self,
      *,
      data_source: dataset_base.RandomAccessDataSource,
      sampler: samplers.Sampler,
      operations: Sequence[transforms.Transformation | ops.Operation] = (),
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

    operations = list(operations)
    for i in range(len(operations)):
      op = operations[i]
      if type(op) is ops.BatchOperation:  # pylint: disable=unidiomatic-typecheck
        operations[i] = transforms.Batch(
            batch_size=op.batch_size,
            drop_remainder=op.drop_remainder,
            batch_fn=op.batch_fn,
        )

    worker_count = _determine_worker_count(worker_count)

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
    self._use_native_dataset_checkpointing = any(
        isinstance(op, transforms.FlatMap) for op in self._operations
    )
    self._state_repr_cache = _StateReprCache()
    self._dataset = self._create_dataset()

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
    return self._local_num_workers * self._shard_options.shard_count  # pyrefly: ignore[missing-attribute]

  def _create_dataset(self) -> dataset.IterDataset:
    """Returns the dataset for this data loader."""
    ds = _SamplerMapDataset(
        self._data_source, self._sampler, self._shard_options  # pyrefly: ignore[bad-argument-type]
    )
    ds = ds.to_iter_dataset(self._read_options)
    for operation in self._operations:
      ds = _apply_transform_to_dataset(operation, ds)
    if self.multiprocessing_options.num_workers > 0 and isinstance(
        ds, dataset_base.SupportsSharedMemoryOutput
    ):
      ds.enable_shared_memory_output()
    ds = ds.map(_ExtractDataMapAdapter())
    if self.multiprocessing_options.num_workers > 0:
      ds = ds.mp_prefetch(self.multiprocessing_options)
    if not self._use_native_dataset_checkpointing:
      ds = _DataLoaderStateIterDataset(
          ds,
          self._shard_options,  # pyrefly: ignore[bad-argument-type]
          self._multiprocessing_options.num_workers,
          self._sampler,
          self._data_source,
          repr_cache=self._state_repr_cache,
      )
    return _DataLoaderPipelineMarkerIterDataset(ds)

  def __iter__(self) -> DataLoaderIterator:
    return DataLoaderIterator(
        self,
        state=None,
        validate_state=not self._use_native_dataset_checkpointing,
    )

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
    expected_sampler = self._state_repr_cache.get_sampler_repr(self._sampler)
    if state[_SAMPLER] != expected_sampler:
      raise ValueError(
          "Sampler in checkpoint does not match dataloader sampler.\n"
          f"sampler in checkpoint: {state[_SAMPLER]}\n"
          f"sampler in dataloader: {expected_sampler}\n"
          "Grain uses `repr(sampler)` to validate the sampler, so you "
          "may need to implement a custom `__repr__`."
      )
    expected_source = self._state_repr_cache.get_data_source_repr(
        self._data_source
    )
    if state[_DATA_SOURCE] != expected_source:
      raise ValueError(
          "DataSource in checkpoint does not match datasource in dataloader.\n"
          f"data source in checkpoint: {state[_DATA_SOURCE]}\n"
          f"data source in dataloader: {expected_source}\n"
          "Grain uses `repr(data_source)` to validate the source, so you "
          "may need to implement a custom `__repr__`."
      )


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

  def __init__(
      self,
      data_loader: DataLoader,
      state: _IteratorState | None,
      validate_state: bool = True,
  ):
    self._data_loader = data_loader
    self._validate_state = validate_state
    if state is not None and self._validate_state:
      self._data_loader._validate_state(state)
    self._iterator = data_loader._dataset.__iter__()  # pylint: disable=protected-access
    if state is not None:
      self._iterator.set_state(state)

  def __iter__(self) -> DataLoaderIterator[_T]:
    return self

  def __next__(self) -> _T:
    start_time = time.perf_counter_ns()
    try:
      return next(self._iterator)
    finally:
      _iterator_get_next_metric.Record(time.perf_counter_ns() - start_time)

  def get_state(self) -> bytes:
    state = self._iterator.get_state()
    return json.dumps(state, indent=4).encode()

  def set_state(self, state: bytes):
    """Sets the state for the underlying iterator.

    Note that state is an implementation detail and can change in the future.
    Args:
      state: state to restore the underlying iterator to.
    """
    state = json.loads(state.decode())
    if self._validate_state:
      self._data_loader._validate_state(state)  # pylint: disable=protected-access  # pyrefly: ignore[bad-argument-type]
    self._iterator.set_state(state)  # pyrefly: ignore[bad-argument-type]

  def start_prefetch(self):
    """Starts processing elements asynchronously in the background.

    Useful when the iterator can be created in advance but the elements are not
    needed immediately. For instance, when recovering iterator and model from a
    checkpoint, recover the iterator first, call ``start_prefech`` and then
    recover the model. This way the time to get the first batch from the
    iterator will be partially or fully hidden behind the time it takes to
    recover the model.

    This method is idempotent and safe to call multiple times.
    """
    self._iterator.start_prefetch()

  ### BEGIN Orbax checkpointing API.
  # See orbax.checkpoint.v1.handlers.StatefulCheckpointable for more details.
  # See https://orbax.readthedocs.io/en/latest/ for usage examples.

  async def save(
      self, directory: checkpoint_base.PathAwaitingCreation
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
    return checkpoint_base.background_save(directory, state)

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

    return checkpoint_base.background_load(directory, set_state_fn)

  ### END Orbax checkpointing API.

  def __str__(self):
    return f"PyGrainDatasetIterator(state={self.get_state().decode()})"


def _source_repr(source: dataset_base.RandomAccessDataSource) -> str:
  """Returns a string representation of the source."""
  # If the source has data in memory avoid printing the data itself.
  if isinstance(source, (list, tuple, np.ndarray)):
    return str(type(source))
  return repr(source)


class _FlatMapAdapter(transforms.FlatMap):
  """Data loader adapter to pass through correct metadata."""

  __slots__ = ("_transform", "max_fan_out")

  def __init__(self, transform: transforms.FlatMap):
    self._transform = transform
    self.max_fan_out = transform.max_fan_out

  def flat_map(self, element: record.Record) -> Sequence[record.Record]:
    value = element.data
    split_offset = element.metadata.index * self.max_fan_out
    result = []
    for split_index, split_value in enumerate(self._transform.flat_map(value)):
      result.append(
          record.Record(
              metadata=record.RecordMetadata(index=split_offset + split_index),
              data=split_value,
          )
      )
    return result


@dataclasses.dataclass(slots=True, repr=False, eq=False)
class _ExtractDataMapAdapter(transforms.Map):
  """Extracts `data` payload from `record.Record` elements."""

  def map(self, element: record.Record) -> Any:
    return element.data


@dataclasses.dataclass(slots=True, repr=False, eq=False)
class _MapTransformAdapter(transforms.Map):
  """Applies a `transforms.Map` to `record.Record.data`."""

  transform: transforms.Map

  def map(self, element: record.Record) -> record.Record:
    return record.Record(
        metadata=element.metadata, data=self.transform.map(element.data)
    )

  def __str__(self) -> str:
    return f"_MapTransformAdapter({type(self.transform).__name__})"


@dataclasses.dataclass(slots=True, repr=False, eq=False)
class _RandomMapTransformAdapter(transforms.Map):
  """Applies a `transforms.RandomMap` using `RecordMetadata.rng`."""

  transform: transforms.RandomMap

  def map(self, element: record.Record) -> record.Record:
    return record.Record(
        metadata=element.metadata,
        data=self.transform.random_map(element.data, element.metadata.rng),  # pyrefly: ignore[bad-argument-type]
    )

  def __str__(self) -> str:
    return f"_RandomMapTransformAdapter({type(self.transform).__name__})"


@dataclasses.dataclass(slots=True, repr=False, eq=False)
class _TfRandomMapTransformAdapter(transforms.Map):
  """Applies a `transforms.TfRandomMap` using `RecordMetadata.rng`."""

  transform: transforms.TfRandomMap

  def map(self, element: record.Record) -> record.Record:
    return record.Record(
        metadata=element.metadata,
        data=self.transform.np_random_map(element.data, element.metadata.rng),  # pyrefly: ignore[bad-argument-type]
    )

  def __str__(self) -> str:
    return f"_TfRandomMapTransformAdapter({type(self.transform).__name__})"


@dataclasses.dataclass(slots=True, repr=False, eq=False)
class _FilterTransformAdapter(transforms.Filter):
  """Applies a `transforms.Filter` to `record.Record.data`."""

  transform: transforms.Filter

  def filter(self, element: record.Record) -> bool:
    return self.transform.filter(element.data)


class _RecordBatchFn:
  """A callable class to batch `record.Record` objects.

  This class wraps a user-provided or default batching function to apply it
  to the `data` field of `record.Record` instances, while preserving metadata.
  """

  __slots__ = ("values_batch_fn",)

  def __init__(self, values_batch_fn: Callable[[Sequence[Any]], Any]):
    self.values_batch_fn = values_batch_fn

  def __call__(self, records: Sequence[record.Record[_T]]) -> record.Record[_T]:
    values = []
    last_metadata = records[0].metadata
    for r in records:
      last_metadata = r.metadata
      values.append(r.data)
    batch = self.values_batch_fn(values)
    return record.Record(metadata=last_metadata.remove_record_key(), data=batch)

  def enable_shared_memory_output(self):
    if self.values_batch_fn is batch_ds.make_batch:
      self.values_batch_fn = functools.partial(
          batch_ds.make_batch, output_to_shared_memory=True
      )
    elif hasattr(self.values_batch_fn, "enable_shared_memory_output"):
      self.values_batch_fn.enable_shared_memory_output()


def _apply_transform_to_dataset(
    transform: transforms.Transformation | ops.Operation,
    ds: dataset.IterDataset,
) -> dataset.IterDataset:
  """Applies the `transform` to the dataset."""
  if isinstance(transform, transforms.Map):
    return ds.map(_MapTransformAdapter(transform))
  elif isinstance(transform, transforms.RandomMap):
    return ds.map(_RandomMapTransformAdapter(transform))
  elif isinstance(transform, transforms.TfRandomMap):
    return ds.map(_TfRandomMapTransformAdapter(transform))
  elif isinstance(transform, transforms.FlatMap):
    return flatmap.FlatMapIterDataset(ds, _FlatMapAdapter(transform))
  elif isinstance(transform, transforms.Filter):
    return ds.filter(_FilterTransformAdapter(transform))
  elif isinstance(transform, transforms.Batch):
    values_batch_fn = transform.batch_fn or batch_ds.make_batch
    batch_fn = _RecordBatchFn(values_batch_fn)

    return ds.batch(
        batch_size=transform.batch_size,
        drop_remainder=transform.drop_remainder,
        batch_fn=batch_fn,
    )
  else:
    # Handle legacy operations
    return _OperationIterDataset(ds, transform)  # pyrefly: ignore[bad-argument-type]
