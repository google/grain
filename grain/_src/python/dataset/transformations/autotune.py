# Copyright 2026 Google LLC
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

"""Autotune Python library for Grain."""

from __future__ import annotations

import copy
import dataclasses
from multiprocessing import queues
import queue
import time
from typing import Any, Callable, Generic, TypeVar, cast

from absl import logging
from concurrent import futures
from grain._src.core import monitoring as grain_monitoring
from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import batch
from grain._src.python.dataset.transformations import filter as filter_transform
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import map as map_transform
from grain._src.python.dataset.transformations import prefetch as prefetch_transform
from grain._src.python.dataset.transformations import process_prefetch
from grain._src.python.dataset.transformations import slice as slice_transform
from grain._src.python.experimental.autotune.python.bindings import AsyncFixedRatioNode
from grain._src.python.experimental.autotune.python.bindings import AutotuneModel
from grain._src.python.experimental.autotune.python.bindings import AutotuneModelConfig
from grain._src.python.experimental.autotune.python.bindings import AutotuneNode
from grain._src.python.experimental.autotune.python.bindings import AutotuneParameter
from grain._src.python.experimental.autotune.python.bindings import BufferRegularizationMode
from grain._src.python.experimental.autotune.python.bindings import ConcurrencyRegularizationMode
from grain._src.python.experimental.autotune.python.bindings import FixedRatioNode
from grain._src.python.experimental.autotune.python.bindings import InterleaveNode
from grain._src.python.experimental.autotune.python.bindings import recover_autotune_node
from grain._src.python.experimental.autotune.python.bindings import SourceNode
from grain._src.python.experimental.autotune.python.bindings import UnknownRatioNode

T = TypeVar("T")

# Stats will be reported from the worker process every
# AUTOTUNE_STATS_REPORTING_FREQ __next__ calls to the worker process iterator.
_AUTOTUNE_STATS_REPORTING_FREQ = 100


def get_autotune_parameter(
    parameter_option: int | AutotuneParameter,
) -> tuple[int, AutotuneParameter | None]:
  """If autotuning is enabled, returns initial value and AutotuneParameter."""
  parameter = None
  if isinstance(parameter_option, AutotuneParameter):
    initial_value = int(parameter_option)
    parameter = parameter_option
  else:
    initial_value = int(parameter_option)
  return initial_value, parameter


def _create_autotune_node(
    ds_iter: dataset.DatasetIterator | dataset.MapDataset,
    allow_unknown_nodes: bool = False,
    concurrency_reg_mode: ConcurrencyRegularizationMode = ConcurrencyRegularizationMode.LINEAR,
    concurrency_reg_weight: float = 0.0,
) -> AutotuneNode:
  """Creates the appropriate AutotuneNode for a dataset."""
  node_type_name = type(ds_iter).__name__
  if isinstance(ds_iter, prefetch_transform.PrefetchDatasetIterator):
    if ds_iter._target_prefetch_buffer_size > 0:  # pylint: disable=protected-access
      node = AsyncFixedRatioNode(
          name=f"{node_type_name}:AsyncFixedRatioNode(input_ratio=1.0)",
          input_ratio=1.0,
          buffer_size=ds_iter.autotune_buffer_size
          if ds_iter.autotune_buffer_size is not None
          else float(ds_iter._target_prefetch_buffer_size),  # pylint: disable=protected-access
          concurrency=ds_iter.autotune_num_threads
          if ds_iter.autotune_num_threads is not None
          else float(ds_iter._target_num_threads),  # pylint: disable=protected-access
      )
      node.set_buffer_regularization(BufferRegularizationMode.BARRIER, 0.05)
      node.set_concurrency_regularization(
          concurrency_reg_mode, concurrency_reg_weight
      )
      return node
    else:
      # If the prefetch buffer size is 0, the node behaves like a sync node.
      return FixedRatioNode(
          name=f"{node_type_name}:FixedRatioNode(input_ratio=1.0)",
          input_ratio=1.0,
      )
  if isinstance(ds_iter, prefetch_transform.ThreadPrefetchDatasetIterator):
    node = AsyncFixedRatioNode(
        name=f"{node_type_name}:AsyncFixedRatioNode(input_ratio=1.0)",
        input_ratio=1.0,
        buffer_size=ds_iter.autotune_buffer_size
        if ds_iter.autotune_buffer_size is not None
        else float(ds_iter._target_prefetch_buffer_size),  # pylint: disable=protected-access
        concurrency=1.0,
    )
    node.set_buffer_regularization(BufferRegularizationMode.BARRIER, 0.05)
    node.set_concurrency_regularization(
        concurrency_reg_mode, concurrency_reg_weight
    )
    return node
  if isinstance(ds_iter, process_prefetch.ProcessPrefetchDatasetIterator):  # pylint: disable=protected-access
    node = AsyncFixedRatioNode(
        name=f"{node_type_name}:AsyncFixedRatioNode(input_ratio=1.0)",
        input_ratio=1.0,
        buffer_size=float(ds_iter.buffer_size),
        concurrency=1.0,
    )
    node.set_buffer_regularization(BufferRegularizationMode.BARRIER, 0.05)
    node.set_concurrency_regularization(
        concurrency_reg_mode, concurrency_reg_weight
    )
    return node
  if isinstance(ds_iter, interleave.TunableInterleaveDatasetIterator):
    return InterleaveNode(
        name=f"{ds_iter}:InterleaveNode",
        cycle_length=ds_iter._autotune_cycle_length  # pylint: disable=protected-access
        if ds_iter._autotune_cycle_length is not None  # pylint: disable=protected-access
        else float(ds_iter._cycle_length),  # pylint: disable=protected-access
        make_iter_buffer_size=float(ds_iter._make_iter_buffer_size),  # pylint: disable=protected-access
    )
  if isinstance(ds_iter, interleave.InterleaveDatasetIterator):
    return InterleaveNode(
        name=f"{ds_iter}:InterleaveNode",
        cycle_length=float(ds_iter._cycle_length),  # pylint: disable=protected-access
        make_iter_buffer_size=float(ds_iter._make_iter_buffer_size),  # pylint: disable=protected-access
    )
  if not ds_iter._parents:  # pylint: disable=protected-access
    return SourceNode(name=f"{node_type_name}:SourceNode")
  if isinstance(ds_iter, map_transform._MapDatasetIterator):  # pylint: disable=protected-access
    return FixedRatioNode(
        name=f"{node_type_name}:FixedRatioNode(input_ratio=1.0)",
        input_ratio=1.0,
    )
  if isinstance(ds_iter, batch._BatchDatasetIterator):  # pylint: disable=protected-access
    return FixedRatioNode(
        name=f"{node_type_name}:FixedRatioNode(input_ratio={ds_iter._batch_size})",  # pylint: disable=protected-access
        input_ratio=ds_iter._batch_size,  # pylint: disable=protected-access
    )
  if isinstance(ds_iter, filter_transform._FilterDatasetIterator):  # pylint: disable=protected-access
    return UnknownRatioNode(name=f"{node_type_name}:UnknownRatioNode")
  if isinstance(ds_iter, map_transform.MapMapDataset):
    return FixedRatioNode(
        name=f"{node_type_name}:FixedRatioNode(input_ratio=1.0)",
        input_ratio=1.0,
    )
  if isinstance(ds_iter, batch.BatchMapDataset):
    return FixedRatioNode(
        name=f"{node_type_name}:FixedRatioNode(input_ratio={ds_iter._batch_size})",  # pylint: disable=protected-access
        input_ratio=ds_iter._batch_size,  # pylint: disable=protected-access
    )
  if isinstance(ds_iter, slice_transform.SliceMapDataset):
    return FixedRatioNode(
        name=f"{node_type_name}:FixedRatioNode(input_ratio=1.0)",
        input_ratio=1.0,
    )
  if allow_unknown_nodes:
    return UnknownRatioNode(name=f"{node_type_name}:UnknownRatioNode")
  raise ValueError(
      f"Unsupported dataset type for autotuning: {type(ds_iter)}. Please file a"
      " bug."
  )


def _build_autotune_model(
    ds_iter: dataset.DatasetIterator | dataset.MapDataset,
    allow_unknown_nodes: bool = False,
    concurrency_reg_mode: ConcurrencyRegularizationMode = ConcurrencyRegularizationMode.LINEAR,
    concurrency_reg_weight: float = 0.0,
) -> AutotuneNode:
  """Builds the autotune model for the given dataset.

  Args:
    ds_iter: A dataset iterator to build the model for.
    allow_unknown_nodes: Whether to allow unknown nodes in the autotune model;
      that is nodes not explicitly supported by autotune will be replaced with a
      black box timing node.
    concurrency_reg_mode: Concurrency regularization mode for async nodes.
    concurrency_reg_weight: Concurrency regularization weight for async nodes.

  Returns:
    The autotune node for the given dataset.
  """
  node = _create_autotune_node(
      ds_iter,
      allow_unknown_nodes,
      concurrency_reg_mode=concurrency_reg_mode,
      concurrency_reg_weight=concurrency_reg_weight,
  )

  if isinstance(
      ds_iter,
      (
          interleave.InterleaveDatasetIterator,
          interleave.TunableInterleaveDatasetIterator,
      ),
  ):
    for parent in ds_iter._iterators_in_use:  # pylint: disable=protected-access
      if parent is not None:
        parent_node = _build_autotune_model(
            parent,
            allow_unknown_nodes,
            concurrency_reg_mode=concurrency_reg_mode,
            concurrency_reg_weight=concurrency_reg_weight,
        )
        node.add_input(parent_node)

  if isinstance(ds_iter, prefetch_transform.PrefetchDatasetIterator):
    input_node = _build_autotune_model(
        ds_iter._map_parent,  # pylint: disable=protected-access
        allow_unknown_nodes,
        concurrency_reg_mode=concurrency_reg_mode,
        concurrency_reg_weight=concurrency_reg_weight,
    )
    node.add_input(input_node)
    return node

  for parent in ds_iter._parents:  # pylint: disable=protected-access
    input_node = _build_autotune_model(
        parent,  # pyrefly: ignore[bad-argument-type]
        allow_unknown_nodes,
        concurrency_reg_mode=concurrency_reg_mode,
        concurrency_reg_weight=concurrency_reg_weight,
    )
    node.add_input(input_node)
  return node


def _record_pipeline_stats(stats: dict[str, Any]):
  """Records USL coeffs and throughput for all nodes in the pipeline."""
  node_name = stats["name"]
  grain_monitoring.record_autotune_usl_coeff(
      node_name, "base_throughput", stats["base_throughput"]
  )
  grain_monitoring.record_autotune_usl_coeff(
      node_name, "contention", stats["contention"]
  )
  grain_monitoring.record_autotune_usl_coeff(
      node_name, "coherency", stats["coherency"]
  )
  output_time_ms = stats["output_time_ms"]
  if output_time_ms > 0:
    grain_monitoring.record_autotune_node_throughput(
        node_name, 1000.0 / output_time_ms
    )

  for param_name, param_value in stats.get("tunable_parameters", {}).items():
    grain_monitoring.record_autotune_parameter(
        node_name, param_name, param_value
    )

  for input_stats in stats.get("inputs", []):
    _record_pipeline_stats(input_stats)


@dataclasses.dataclass
class _WarmupState:
  """State for autotuning warmup."""

  warmup_steps_remaining: int


_BYTE_SIZE_SAMPLE_INTERVAL = 50


class _AutotuneDatasetIterator(dataset.DatasetIterator[T]):
  """Iterator that wraps a PyGrain dataset iterator and its AutotuneNode.

  This class is a proxy class and designates method calls to the wrapped inner
  dataset iterator.
  """

  def __init__(
      self,
      inner: dataset.DatasetIterator[T],
      autotune_node: AutotuneNode,
      warmup_state: _WarmupState,
      output_node: AutotuneNode | None = None,
      model: AutotuneModel | None = None,
      optimization_frequency: int = 100,
      autotune_stats_queue: queues.Queue[bytes] | None = None,
  ):
    self._autotune_node = autotune_node
    self._warmup_state = warmup_state
    self._output_node = output_node
    self._model = model
    self._optimization_frequency = optimization_frequency
    self._next_calls_since_last_report = _AUTOTUNE_STATS_REPORTING_FREQ - 1
    self._autotune_stats_queue = autotune_stats_queue
    self._element_count = 0
    self._last_element_size_bytes = 0

    if isinstance(inner, prefetch_transform.PrefetchDatasetIterator):
      inputs = autotune_node.get_inputs()
      assert len(inputs) == 1
      if inputs:
        map_node = inputs[0]
        inner._map_parent = _AutotuneMapDataset(
            inner._map_parent, map_node, warmup_state, output_node=autotune_node
        )
    elif isinstance(
        inner,
        (
            interleave.InterleaveDatasetIterator,
            interleave.TunableInterleaveDatasetIterator,
        ),
    ):
      wrapped_iterators = []
      inputs = autotune_node.get_inputs()
      input_idx = 0
      for it in inner._iterators_in_use:
        if it is not None:
          input_node = inputs[input_idx]
          input_idx += 1
          new_it = _AutotuneDatasetIterator(
              it,
              input_node,
              warmup_state,
              output_node=autotune_node,
              model=None,
          )
          new_it._ctx.autotuning_enabled = True
          new_it._ctx.autotuning_allow_unknown_nodes = (
              inner._ctx.autotuning_allow_unknown_nodes
          )
          wrapped_iterators.append(new_it)
        else:
          wrapped_iterators.append(None)
      inner._iterators_in_use = wrapped_iterators
    else:
      wrapped_parents = []
      for parent, input_node in zip(inner._parents, autotune_node.get_inputs()):
        new_parent = _AutotuneDatasetIterator(
            parent,
            input_node,
            warmup_state,
            output_node=autotune_node,
            model=None,
        )
        wrapped_parents.append(new_parent)
      inner._parents = tuple(wrapped_parents)
    # The original iterator.
    self._inner = inner

    if isinstance(
        inner, prefetch_transform.ThreadPrefetchDatasetIterator
    ) and isinstance(inner._maybe_nonnative_parent, dataset.DatasetIterator):
      # Thread prefetch uses _maybe_nonnative_parent to refer to the parent
      # iterator. We need to make sure it uses the wrapped parent iterator.
      inner._maybe_nonnative_parent = inner._parent

    # Prefetchers are special cases; we need to make sure futures are wrapped
    # so we can measure stall. If the prefetch buffer size is 0, no proxy will
    # be installed.
    if isinstance(self._inner, prefetch_transform.PrefetchDatasetIterator):
      self._inner.set_executor_wrapper(
          lambda executor: _AutotuneExecutorProxy(  # pyrefly: ignore[bad-argument-type]
              executor, self._autotune_node, self._warmup_state
          )
      )
      if hasattr(self._inner, "_executor"):
        self._inner._executor = _AutotuneExecutorProxy(  # pylint: disable=protected-access  # pyrefly: ignore[bad-assignment]
            self._inner._executor, self._autotune_node, self._warmup_state
        )

    # TODO: b/482163919 - Add support for other async iterators.

  def _maybe_optimize(self) -> None:
    """Runs autotune optimization and records stats."""
    if self._model:
      try:
        start_optimize = time.perf_counter()
        if self._model.maybe_optimize(self._autotune_node):
          duration_ms = (time.perf_counter() - start_optimize) * 1000
          grain_monitoring.record_autotune_optimization_latency(duration_ms)
          _record_pipeline_stats(self._autotune_node.get_pipeline_stats())
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            "PyGrain Autotune optimization failed: %s. Continuing "
            "pipeline execution normally with last safe parameters.",
            e,
        )

  def __getattr__(self, name: str) -> Any:
    """Delegate all attribute access to the inner iterator if not defined."""
    return getattr(self._inner, name)

  def _sync_worker_stats(self) -> None:
    if isinstance(
        self._inner, process_prefetch.ProcessPrefetchDatasetIterator  # pylint: disable=protected-access
    ):
      autotune_stats_queue = self._inner._autotune_stats_queue  # pylint: disable=protected-access
      if autotune_stats_queue is not None:
        snapshot_bytes = None
        try:
          while True:
            snapshot_bytes = autotune_stats_queue.get_nowait()
        except queue.Empty:
          pass
        if snapshot_bytes is not None:
          node = recover_autotune_node(snapshot_bytes)
          self._autotune_node.clear_inputs()
          self._autotune_node.add_input(node)

  def __next__(self) -> T:
    if self._warmup_state.warmup_steps_remaining > 0:
      result = next(self._inner)
      if self._output_node is None:
        self._warmup_state.warmup_steps_remaining -= 1
      return result

    self._sync_worker_stats()

    # If inputting into an async node, do not pause the consumer timer.
    pause_consumer = self._output_node and not self._output_node.is_async
    try:
      if pause_consumer:
        self._output_node.record_pause()

      # Note: try to execute optimization in a place where timings won't be
      # affected.
      self._maybe_optimize()

      self._autotune_node.record_start()
      result = next(self._inner)
      self._element_count += 1
      if (
          self._element_count % _BYTE_SIZE_SAMPLE_INTERVAL == 1
          or self._last_element_size_bytes == 0
      ):
        self._last_element_size_bytes = tree_lib.estimate_byte_size(result)
      self._autotune_node.record_element_size_bytes(
          self._last_element_size_bytes
      )
      return result
    finally:
      self._autotune_node.record_end()
      self._sync_worker_stats()
      if self._autotune_stats_queue is not None:
        self._next_calls_since_last_report += 1
        if self._next_calls_since_last_report >= _AUTOTUNE_STATS_REPORTING_FREQ:
          snapshot = self._autotune_node.get_snapshot_bytes()
          try:
            self._autotune_stats_queue.put_nowait(snapshot)
            self._next_calls_since_last_report = 0
          except queue.Full:
            try:
              self._autotune_stats_queue.get_nowait()
              self._autotune_stats_queue.put_nowait(snapshot)
              self._next_calls_since_last_report = 0
            except (queue.Empty, queue.Full):
              pass
      if isinstance(
          self._inner,
          (
              interleave.InterleaveDatasetIterator,
              interleave.TunableInterleaveDatasetIterator,
          ),
      ):
        self._autotune_node.clear_inputs()
        for iterator in self._inner._iterators_in_use:  # pylint: disable=protected-access
          if iterator is not None:
            assert isinstance(iterator, _AutotuneDatasetIterator)
            self._autotune_node.add_input(iterator._autotune_node)  # pylint: disable=protected-access
      if pause_consumer:
        self._output_node.record_resume()

  def start_prefetch(self) -> None:
    self._inner.start_prefetch()

  def __str__(self) -> str:
    return f"_AutotuneDatasetIterator({self._inner})"

  def __repr__(self) -> str:
    return f"_AutotuneDatasetIterator({self._inner!r})"

  def get_state(self) -> dict[str, Any]:
    return self._inner.get_state()

  def set_state(self, state: dict[str, Any]) -> None:
    self._inner.set_state(state)

  def close(self) -> None:
    if hasattr(self._inner, "close"):
      self._inner.close()

  def __del__(self):
    try:
      self.close()
    except Exception:  # pylint: disable=broad-except
      pass


class _AutotuneFutureProxy(Generic[T]):
  """Proxy class for a future that wraps an AutotuneNode.

  This is essential for measuring stall in async nodes.
  """

  def __init__(
      self,
      future: futures.Future[T],
      consumer_node: AutotuneNode,
      warmup_state: _WarmupState,
  ):
    self._future = future
    self._consumer_node = consumer_node
    self._warmup_state = warmup_state

  def __getattr__(self, name: str) -> Any:
    """Delegate all attribute access to the inner future if not defined."""
    return getattr(self._future, name)

  def result(self, timeout: float | None = None) -> T:
    """Returns the result of the future after recording the consumer time."""
    if self._warmup_state.warmup_steps_remaining > 0:
      return self._future.result(timeout=timeout)
    self._consumer_node.record_pause()
    try:
      res = self._future.result(timeout=timeout)
      return res
    finally:
      self._consumer_node.record_resume()


class _AutotuneExecutorProxy:
  """Proxy class for an autotune executor."""

  def __init__(
      self,
      executor: futures.Executor,
      autotune_node: AutotuneNode,
      warmup_state: _WarmupState,
  ):
    self._executor = executor
    self._autotune_node = autotune_node
    self._warmup_state = warmup_state

  def __getattr__(self, name: str) -> Any:
    """Delegate all attribute access to the inner executor if not defined."""
    return getattr(self._executor, name)

  def submit(
      self, fn: Callable[..., T], /, *args: Any, **kwargs: Any
  ) -> _AutotuneFutureProxy[T]:
    return _AutotuneFutureProxy(
        self._executor.submit(fn, *args, **kwargs),
        self._autotune_node,
        self._warmup_state,
    )


class AutotuneIterDataset(dataset.IterDataset[T]):
  """Dataset that wraps a PyGrain dataset and its corresponding AutotuneNode."""

  def __init__(
      self,
      parent: dataset.IterDataset[T],
      model_config: AutotuneModelConfig = AutotuneModelConfig(),
      allow_unknown_nodes=False,
      create_model: bool = True,
      autotune_stats_queue: queues.Queue[bytes] | None = None,
      warmup_state: _WarmupState | None = None,
      concurrency_reg_mode: ConcurrencyRegularizationMode = ConcurrencyRegularizationMode.LINEAR,
      concurrency_reg_weight: float = 0.0,
  ):
    super().__init__(parent)
    self._allow_unknown_nodes = allow_unknown_nodes
    self._autotune_node = None
    self._model_config = model_config
    self._create_model = create_model
    self._autotune_stats_queue = autotune_stats_queue
    self._warmup_state = warmup_state
    self._concurrency_reg_mode = concurrency_reg_mode
    self._concurrency_reg_weight = concurrency_reg_weight

  def __iter__(self) -> dataset.DatasetIterator[T]:
    inner_iter = cast(dataset.DatasetIterator[T], iter(self._parent))
    # Lazily build the autotune model the first time the iterator is created.
    if self._autotune_node is None:
      self._autotune_node = _build_autotune_model(
          inner_iter,
          self._allow_unknown_nodes,
          concurrency_reg_mode=self._concurrency_reg_mode,
          concurrency_reg_weight=self._concurrency_reg_weight,
      )

    warmup_state = self._warmup_state
    if warmup_state is None:
      warmup_state = _WarmupState(self._model_config.get_warmup_steps())

    if self._autotune_stats_queue is not None:
      num_processes = inner_iter._ctx.mp_context.process_count
      adjusted_cpu_budget = self._model_config.get_cpu_budget() // num_processes
      adjusted_ram_budget = (
          self._model_config.get_ram_budget_gb() / num_processes
      )

      adjusted_config = AutotuneModelConfig(
          cpu_budget=adjusted_cpu_budget,
          ram_budget_gb=adjusted_ram_budget,
          optimization_frequency=self._model_config.get_optimization_frequency(),
          warmup_steps=warmup_state.warmup_steps_remaining,
      )
      model = AutotuneModel(adjusted_config)
    else:
      model = AutotuneModel(self._model_config) if self._create_model else None

    iterator = _AutotuneDatasetIterator(
        inner_iter,
        self._autotune_node,
        warmup_state,
        output_node=None,
        model=model,
        optimization_frequency=self._model_config.get_optimization_frequency(),
        autotune_stats_queue=self._autotune_stats_queue,
    )
    ctx = iterator._ctx
    ctx.autotuning_enabled = True
    ctx.autotuning_allow_unknown_nodes = self._allow_unknown_nodes
    ctx.autotuning_warmup_state = warmup_state
    if self._create_model:
      ctx.autotuning_model_config_args = {
          "cpu_budget": self._model_config.get_cpu_budget(),
          "ram_budget_gb": self._model_config.get_ram_budget_gb(),
          "optimization_frequency": (
              self._model_config.get_optimization_frequency()
          ),
          "warmup_steps": self._model_config.get_warmup_steps(),
      }
    return iterator

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


class _AutotuneMapDataset(dataset.MapDataset[T]):
  """Dataset that wraps a MapDataset and its AutotuneNode."""

  def __init__(
      self,
      inner: dataset.MapDataset[T],
      autotune_node: AutotuneNode,
      warmup_state: _WarmupState,
      output_node: AutotuneNode | None = None,
  ):
    # Create a shallow copy of the inner dataset to avoid mutating the original
    # when we replace its parents.
    self._inner = copy.copy(inner)
    self._autotune_node = autotune_node
    self._warmup_state = warmup_state
    self._output_node = output_node

    wrapped_parents = []
    for parent, input_node in zip(inner.parents, autotune_node.get_inputs()):
      wrapped_parents.append(
          _AutotuneMapDataset(
              parent, input_node, warmup_state, output_node=self._autotune_node
          )
      )

    self._element_count = 0
    self._last_element_size_bytes = 0

    if wrapped_parents:
      self._inner._parents = tuple(wrapped_parents)

  def __len__(self) -> int:
    return len(self._inner)

  def __getitem__(self, index):
    if self._warmup_state.warmup_steps_remaining > 0:
      return self._inner[index]

    async_output = bool(self._output_node and self._output_node.is_async)
    try:
      if self._output_node and not async_output:
        self._output_node.record_pause()
      self._autotune_node.record_start()
      result = self._inner[index]
      self._element_count += 1
      if (
          self._element_count % _BYTE_SIZE_SAMPLE_INTERVAL == 1
          or self._last_element_size_bytes == 0
      ):
        self._last_element_size_bytes = tree_lib.estimate_byte_size(result)
      self._autotune_node.record_element_size_bytes(
          self._last_element_size_bytes
      )
      return result
    finally:
      self._autotune_node.record_end()
      if self._output_node and not async_output:
        self._output_node.record_resume()

  def __str__(self) -> str:
    return f"_AutotuneMapDataset({self._inner})"

  def __repr__(self) -> str:
    return f"_AutotuneMapDataset({self._inner!r})"


def autotune(
    ds: dataset.IterDataset[T],
    model_config: AutotuneModelConfig = AutotuneModelConfig(),
    allow_unknown_nodes: bool = False,
    concurrency_reg_mode: ConcurrencyRegularizationMode = ConcurrencyRegularizationMode.LINEAR,
    concurrency_reg_weight: float = 0.0,
) -> AutotuneIterDataset[T]:
  """Returns an AutotuneIterDataset for the given dataset.

  Args:
    ds: The dataset to autotune.
    model_config: The autotune model config.
    allow_unknown_nodes: Whether to allow unknown nodes in the autotune model;
      that is nodes not explicitly supported by autotune will be replaced with a
      black box timing node.
    concurrency_reg_mode: Concurrency regularization mode for async nodes.
    concurrency_reg_weight: Concurrency regularization weight for async nodes.

  Returns:
    An AutotuneIterDataset for the given dataset.
  """

  return AutotuneIterDataset(
      ds,
      model_config=model_config,
      allow_unknown_nodes=allow_unknown_nodes,
      concurrency_reg_mode=concurrency_reg_mode,
      concurrency_reg_weight=concurrency_reg_weight,
  )
