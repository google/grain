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
"""PyTorch-aware multiprocess prefetch with thread capping and tensor IPC.

Grain's `mp_prefetch` moves `np.ndarray` leaves from the workers through shared
memory (`shared_memory_array.copy_to_shm`) and pickles all other leaves. To pass
CPU `torch.Tensor`s through shared memory too, `multiprocess_prefetch` in this
module puts Grain's `mp_prefetch` between two datasets. They follow the pattern
of `copy_to_shm` and `open_from_shm`, a leaf function applied with
`tree_lib.map_structure`:

1. In the worker, `_encode_leaf` replaces each tensor with an `_IpcTensor`: the
   NumPy array that shares the tensor's memory (`torch.Tensor.numpy`) and the
   metadata to rebuild the tensor.
2. Grain's `copy_to_shm` moves the NumPy array to shared memory like any other
   array.
3. In the main process, `_decode_leaf` turns each `_IpcTensor` back into a
   tensor that shares its memory with the array.

TODO: Support `torch.Tensor` leaves in `copy_to_shm` and
`open_from_shm`, and remove `_IpcTensor` and the encode/decode datasets below.
"""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import os
from typing import Any, NamedTuple, TypeVar

from grain._src.core import tree_lib
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats as dataset_stats
from grain._src.python.experimental.torch import options as torch_options
import numpy as np

T = TypeVar("T")


# TODO: b/584830326 - Support `torch.Tensor` leaves in
# `shared_memory_array.copy_to_shm` and `open_from_shm`, and remove `_IpcTensor`
# and the encode and decode datasets below.
@dataclasses.dataclass(frozen=True, slots=True)
class _TensorMeta:
  """Opaque PyTree leaf holding metadata to rebuild a tensor from an array.

  Attributes:
    dtype: The tensor's dtype. It differs from the array's dtype for `bfloat16`.
    tensor_cls: The tensor's class: `torch.Tensor` or a subclass of it.
    attrs: The instance attributes of the tensor, if it has any.
  """

  dtype: Any
  tensor_cls: type[Any]
  attrs: dict[str, Any] | None


class _IpcTensor(NamedTuple):
  """A CPU tensor in the form that Grain's `copy_to_shm` moves to shared memory.

  `copy_to_shm` only moves `np.ndarray` leaves, so the tensor is held as the
  NumPy array that shares its memory. `_IpcTensor` is a `NamedTuple` so that
  `copy_to_shm` traverses into it and finds `data`.

  Attributes:
    data: The tensor as a NumPy array of the same dtype and shape. NumPy has no
      `bfloat16`, so a `bfloat16` tensor is held as an `int16` array of its
      bits.
    meta: Metadata to rebuild the tensor from `data`.
  """

  data: np.ndarray
  meta: _TensorMeta


# Methods that customize how the instances of a class are pickled.
_PICKLE_METHODS = (
    "__getstate__",
    "__setstate__",
    "__reduce__",
    "__reduce_ex__",
)


def _keeps_state_in_dict(tensor_cls: type[Any], base_cls: type[Any]) -> bool:
  """Returns whether `__dict__` holds all Python state of `tensor_cls` tensors.

  It does not if `tensor_cls`, or a base of it that `base_cls` does not have,
  declares `__slots__` or customizes pickling.

  Args:
    tensor_cls: The class of a tensor, a subclass of `base_cls`.
    base_cls: `torch.Tensor`, which is imported lazily by the caller.
  """
  for cls in tensor_cls.__mro__:
    if cls in base_cls.__mro__:
      continue
    namespace = vars(cls)
    if namespace.get("__slots__") or any(
        method in namespace for method in _PICKLE_METHODS
    ):
      return False
  return True


def _is_encodable_cpu_tensor(x: Any) -> bool:
  """Returns whether `x` is a tensor that can cross IPC as a NumPy array."""
  import torch  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  if not isinstance(x, torch.Tensor):
    return False
  if x.device.type != "cpu" or x.requires_grad:
    return False
  if (
      x.layout is not torch.strided
      or x.is_quantized
      or x.is_nested
      or getattr(x, "is_meta", False)
  ):
    return False
  tensor_cls = type(x)
  if tensor_cls is torch.Tensor:
    return True
  # A tensor of a subclass is rebuilt from the array, the class and the
  # `__dict__` of the tensor. A subclass that needs more is left to pickle.
  return (
      tensor_cls.__torch_dispatch__ is torch.Tensor.__torch_dispatch__
      and _keeps_state_in_dict(tensor_cls, torch.Tensor)
  )


def _encode_leaf(x: Any) -> Any:
  """Converts eligible CPU `torch.Tensor` leaves into `_IpcTensor` views."""
  try:
    import torch  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  except ImportError:
    return x  # Without PyTorch, `x` cannot be a tensor.
  if not _is_encodable_cpu_tensor(x):
    return x
  tensor_cls = type(x)
  base_t = x if tensor_cls is torch.Tensor else x.as_subclass(torch.Tensor)
  base_t = base_t.resolve_conj().resolve_neg().contiguous()
  # `torch.Tensor.numpy` does not copy: the array shares the tensor's memory.
  if base_t.dtype == torch.bfloat16:
    # NumPy has no `bfloat16`, so the bits are passed as `int16`, which
    # `_decode_leaf` views as `bfloat16` again.
    data = base_t.view(torch.int16).numpy()
  else:
    try:
      data = base_t.numpy()
    except TypeError:
      return x  # NumPy has no equivalent of the dtype, so `x` is pickled.
  attrs = dict(x.__dict__) if hasattr(x, "__dict__") and x.__dict__ else None
  return _IpcTensor(
      data=data,
      meta=_TensorMeta(
          dtype=base_t.dtype,
          tensor_cls=tensor_cls,
          attrs=attrs,
      ),
  )


def _decode_leaf(x: Any) -> Any:
  """Rebuilds a CPU `torch.Tensor` zero-copy from an `_IpcTensor` node."""
  if type(x) is not _IpcTensor:  # pylint: disable=unidiomatic-typecheck
    return x
  import torch  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

  data = x.data
  if not data.flags.writeable:
    data = data.copy()
  meta = x.meta
  tensor = torch.from_numpy(data)
  if tensor.dtype != meta.dtype:
    tensor = tensor.view(meta.dtype)  # `bfloat16` bits passed as `int16`.
  if meta.tensor_cls is not torch.Tensor:
    tensor = tensor.as_subclass(meta.tensor_cls)
  if meta.attrs:
    tensor.__dict__.update(meta.attrs)
  return tensor


def _is_ipc_tensor(x: Any) -> bool:
  return type(x) is _IpcTensor  # pylint: disable=unidiomatic-typecheck


class _TorchIpcEncodeDatasetIterator(dataset.DatasetIterator[Any]):
  """Worker-side iterator encoding CPU tensors into `_IpcTensor` nodes."""

  _MUTATES_ELEMENT_SPEC = False

  @dataset_stats.record_next_duration_if_output
  def __next__(self) -> Any:
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is None:
        return None
      return tree_lib.map_structure(_encode_leaf, element)

  def get_state(self) -> dict[str, Any]:
    return self._parent.get_state()

  def set_state(self, state: dict[str, Any]) -> None:
    self._parent.set_state(state)

  def _get_next_index(self) -> int:
    return dataset.get_next_index(self._parent)

  def _set_next_index(self, next_index: int) -> None:
    dataset.set_next_index(self._parent, next_index)

  def __str__(self) -> str:
    return "_TorchIpcEncodeDatasetIterator()"


class _TorchIpcEncodeIterDataset(dataset.IterDataset[Any]):
  """Worker-side dataset forwarding slicing and shared-memory batching."""

  _MUTATES_ELEMENT_SPEC = False

  def enable_shared_memory_output(self) -> None:
    if isinstance(self._parent, base.SupportsSharedMemoryOutput):
      self._parent.enable_shared_memory_output()

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    dataset.set_slice(self._parent, sl, sequential_slice)

  def __iter__(self) -> dataset.DatasetIterator[Any]:
    return _TorchIpcEncodeDatasetIterator(self._parent.__iter__())

  def __str__(self) -> str:
    return "_TorchIpcEncodeIterDataset()"

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


class _TorchIpcDecodeDatasetIterator(dataset.DatasetIterator[T]):
  """Parent-side iterator reconstructing `torch.Tensor` leaves zero-copy."""

  # The elements of the parent hold `_IpcTensor`s, so this iterator records
  # the specification of its own output for the execution statistics.
  _MUTATES_ELEMENT_SPEC = True

  @dataset_stats.record_next_duration_if_output
  def __next__(self) -> T:
    element = next(self._parent)
    with self._stats.record_self_time():
      if element is None:
        return None  # pyrefly: ignore[bad-return]
      return self._stats.record_output_spec(
          tree_lib.map_structure(_decode_leaf, element, is_leaf=_is_ipc_tensor)
      )

  def get_state(self) -> dict[str, Any]:
    return self._parent.get_state()

  def set_state(self, state: dict[str, Any]) -> None:
    self._parent.set_state(state)

  def _get_next_index(self) -> int:
    return dataset.get_next_index(self._parent)

  def _set_next_index(self, next_index: int) -> None:
    dataset.set_next_index(self._parent, next_index)

  def __str__(self) -> str:
    return "_TorchIpcDecodeDatasetIterator()"


class _TorchIpcDecodeIterDataset(dataset.IterDataset[T]):
  """Parent-side dataset decoding `_IpcTensor` leaves back to tensors."""

  # Same as for `_TorchIpcDecodeDatasetIterator`.
  _MUTATES_ELEMENT_SPEC = True

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _TorchIpcDecodeDatasetIterator(self._parent.__iter__())

  def __str__(self) -> str:
    return "_TorchIpcDecodeIterDataset()"

  @property
  def _element_spec(self) -> Any:
    return dataset.get_element_spec(self._parent)


def _torch_worker_init(
    worker_index: int,
    worker_count: int,
    *,
    intra_op_threads: int | None,
    user_init_fn: Callable[[int, int], None] | None,
) -> None:
  """Caps the worker's PyTorch intra-op threads, then runs `user_init_fn`."""
  if intra_op_threads is not None:
    # The environment variables cover libraries that read them when they load;
    # PyTorch itself may already be imported, so it is also set explicitly.
    threads_str = str(intra_op_threads)
    os.environ["OMP_NUM_THREADS"] = threads_str
    os.environ["MKL_NUM_THREADS"] = threads_str
    try:
      import torch  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
    except ImportError:
      pass  # Without PyTorch, only the environment variables apply.
    else:
      torch.set_num_threads(intra_op_threads)
  if user_init_fn is not None:
    user_init_fn(worker_index, worker_count)


def multiprocess_prefetch(
    ds: dataset.IterDataset[T],
    options: grain_options.MultiprocessingOptions | None = None,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> dataset.IterDataset[T]:
  """Prefetches elements in worker processes with PyTorch thread cap & IPC.

  1. Caps intra-op threads per worker by calling ``torch.set_num_threads``
     (and setting ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``) with
     ``options.worker_intra_op_threads``, defaulting to ``1``, before invoking
     ``worker_init_fn``. This matches ``torch.utils.data.DataLoader`` workers.
  2. Routes plain CPU ``torch.Tensor`` leaves through Grain's
     ``SharedMemoryArray`` IPC as zero-copy NumPy views so
     ``DatasetOptions.min_shm_size`` applies uniformly to tensors and NumPy
     arrays while eliminating per-tensor file descriptors and shutdown errors.

  Args:
    ds: Parent ``IterDataset`` to prefetch from.
    options: Multiprocessing options (accepts either
      ``grain.experimental.torch.MultiprocessingOptions`` or
      ``grain.MultiprocessingOptions``; defaults to 10 workers with 1 intra-op
      thread per worker).
    worker_init_fn: Optional callback ``fn(worker_index, worker_count)`` invoked
      in each spawned worker after the thread cap is applied.
    sequential_slice: Whether each worker processes a contiguous slice.

  Returns:
    An ``IterDataset`` prefetching elements across worker processes.

  Raises:
    ValueError: If ``worker_intra_op_threads`` is non-positive.
  """
  if options is None:
    options = torch_options.MultiprocessingOptions(num_workers=10)
  intra_op_threads = getattr(options, "worker_intra_op_threads", 1)
  if intra_op_threads is not None and intra_op_threads <= 0:
    raise ValueError(
        "`worker_intra_op_threads` must be a positive integer or None, got"
        f" {intra_op_threads}."
    )
  if options.num_workers == 0 or dataset.is_in_free_threaded_python():
    return ds.mp_prefetch(
        options=options,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    )

  effective_worker_init_fn = worker_init_fn
  if intra_op_threads is not None:
    effective_worker_init_fn = functools.partial(
        _torch_worker_init,
        intra_op_threads=intra_op_threads,
        user_init_fn=worker_init_fn,
    )
  encoded_ds = _TorchIpcEncodeIterDataset(ds)
  prefetched_ds = encoded_ds.mp_prefetch(
      options=options,
      worker_init_fn=effective_worker_init_fn,
      sequential_slice=sequential_slice,
  )
  return _TorchIpcDecodeIterDataset(prefetched_ds)
