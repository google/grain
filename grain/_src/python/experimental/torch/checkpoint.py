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
"""Checkpointing wrapper for PyTorch distributed training.

``IteratorCheckpointWrapper`` with ``torch.distributed.checkpoint`` (DCP) is
the supported way to checkpoint Grain iterators across ``torch.distributed``
ranks. Grain's Orbax integration (``DatasetIterator.save`` / ``load`` and
``grain.checkpoint``) names per-process files by JAX process index and is not
supported with ``torch.distributed`` ranks.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

from grain._src.python.experimental.torch import distributed as torch_distributed

_STATE_DICT_VERSION = 1


class IteratorCheckpointWrapper:
  """Wraps a Grain iterator to implement DCP's ``Stateful`` protocol.

  Implements ``torch.distributed.checkpoint.stateful.Stateful``
  (``state_dict`` / ``load_state_dict``) on top of Grain's ``get_state`` /
  ``set_state``.

  ``torch.distributed.checkpoint`` (DCP) keeps a single copy of each
  non-tensor state-dict key across ranks on save, so returning ``get_state()``
  directly from ``state_dict()`` would make every rank restore one rank's
  progress. ``IteratorCheckpointWrapper`` stores each rank's serialized state
  under ``rank_{i}`` (scoped to ``group`` when DP is a subgroup of the global
  world, such as under tensor parallelism).

  Saving from more than one rank with DCP requires PyTorch 2.4 or later.
  Earlier releases write non-tensor values only from the coordinator rank, so
  the state of the other ranks is not saved and cannot be restored.

  Iterators of ``grain.experimental.torch`` datasets
  (``grain.experimental.torch.DatasetIterator``) implement ``Stateful`` on top
  of this class, so they can be passed to ``dcp.save`` and ``dcp.load``
  directly. Use ``IteratorCheckpointWrapper`` explicitly in two cases:

  - To pass ``group``. The iterators above always use the default process
    group.
  - To checkpoint an iterator of a ``grain.MapDataset`` or
    ``grain.IterDataset``, which does not implement ``Stateful``.

  ``IteratorCheckpointWrapper`` holds only a reference to the iterator:
  ``state_dict()`` reads the iterator's state when it is called, and nothing is
  cached. Create it at each save or load call, as in the example below, so that
  it always wraps the iterator in use. An instance that is kept around still
  wraps the old iterator after the iterator is replaced.

  Example:
    An iterator of a ``grain.experimental.torch`` dataset needs no wrapper::

      import torch.distributed.checkpoint as dcp
      from grain.experimental import torch as grain_torch

      it = iter(grain_torch.MapDataset.range(8).to_iter_dataset())
      dcp.save({"data": it}, checkpoint_id=path)
      dcp.load({"data": it}, checkpoint_id=path)

    A process group other than the default one (here ``dp_group``) is passed
    through the wrapper::

      dcp.save(
          {"data": grain_torch.IteratorCheckpointWrapper(it, group=dp_group)},
          checkpoint_id=path,
      )
      dcp.load(
          {"data": grain_torch.IteratorCheckpointWrapper(it, group=dp_group)},
          checkpoint_id=path,
      )

    An iterator of a ``grain.MapDataset`` or ``grain.IterDataset`` always needs
    the wrapper, with or without ``group``::

      plain_it = iter(grain.MapDataset.range(8).to_iter_dataset())
      dcp.save(
          {"data": grain_torch.IteratorCheckpointWrapper(plain_it)},
          checkpoint_id=path,
      )
      dcp.load(
          {"data": grain_torch.IteratorCheckpointWrapper(plain_it)},
          checkpoint_id=path,
      )
  """

  def __init__(self, iterator: Any, *, group: Any = None):
    if not (hasattr(iterator, "get_state") and hasattr(iterator, "set_state")):
      raise TypeError(
          "`IteratorCheckpointWrapper` expects an iterator implementing"
          f" `get_state` and `set_state`, got {type(iterator).__name__}."
      )
    self._iterator = iterator
    self._group = group

  def state_dict(self) -> dict[str, Any]:
    """Returns a per-rank dictionary suitable for ``dcp.save``."""
    rank, count = torch_distributed.get_process_index_and_count(self._group)
    raw_state = self._iterator.get_state()
    if isinstance(raw_state, (bytes, bytearray)):
      payload: str | bytes = bytes(raw_state)
    else:
      payload = json.dumps(raw_state)
    return {
        "version": _STATE_DICT_VERSION,
        "count": count,
        f"rank_{rank}": payload,
    }

  def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
    """Restores this rank's iterator progress from ``state_dict``."""
    version = state_dict.get("version")
    if version != _STATE_DICT_VERSION:
      raise ValueError(
          f"Unsupported `IteratorCheckpointWrapper` version: {version!r}"
          f" (expected {_STATE_DICT_VERSION})."
      )
    rank, count = torch_distributed.get_process_index_and_count(self._group)
    saved_count = state_dict.get("count")
    if saved_count != count:
      raise ValueError(
          "Checkpoint shard count mismatch: checkpoint was saved with"
          f" {saved_count} ranks, but current group has {count} ranks."
      )
    rank_key = f"rank_{rank}"
    if rank_key not in state_dict:
      raise ValueError(
          f"Missing checkpoint state for {rank_key!r} in `state_dict`."
      )
    payload = state_dict[rank_key]
    if isinstance(payload, str):
      self._iterator.set_state(json.loads(payload))
    elif isinstance(payload, (bytes, bytearray)):
      self._iterator.set_state(bytes(payload))
    else:
      raise TypeError(
          f"Unexpected checkpoint payload type for {rank_key!r}:"
          f" {type(payload).__name__}."
      )
