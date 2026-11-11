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
"""Checkpointing adapters and helpers for PyTorch distributed training."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import json
from typing import Any

from etils import epath
from grain._src.python.checkpoint import base as checkpoint_base
from grain._src.python.experimental.torch import distributed as torch_distributed

_STATE_DICT_VERSION = 1


class IteratorState:
  """``torch.distributed.checkpoint`` ``Stateful`` adapter for Grain iterators.

  ``torch.distributed.checkpoint`` (DCP) deduplicates identical state-dict keys
  across ranks on save, so a plain ``state_dict = get_state`` alias would
  overwrite every rank's progress with rank 0's state. ``IteratorState``
  stores each rank's serialized state under ``rank_{i}`` (scoped to ``group``
  when DP is a subgroup of the global world, such as under tensor parallelism).
  """

  def __init__(self, iterator: Any, *, group: Any = None):
    if not (hasattr(iterator, "get_state") and hasattr(iterator, "set_state")):
      raise TypeError(
          "`IteratorState` expects an iterator implementing `get_state`"
          f" and `set_state`, got {type(iterator).__name__}."
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
          f"Unsupported `IteratorState` version: {version!r} (expected"
          f" {_STATE_DICT_VERSION})."
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


async def background_save(
    directory: checkpoint_base.PathAwaitingCreation,
    state: str,
    group: Any = None,
) -> None:
  """Asynchronously writes iterator state named by PyTorch distributed rank."""
  resolved_dir = await directory.await_creation()
  process_index, process_count = torch_distributed.get_process_index_and_count(
      group
  )
  filename = resolved_dir / f"process_{process_index}-of-{process_count}.json"
  await asyncio.to_thread(filename.write_text, state)


async def background_load(
    directory: epath.Path,
    set_state_fn: Callable[[str], None],
    group: Any = None,
) -> None:
  """Asynchronously reads iterator state named by PyTorch distributed rank."""
  process_index, process_count = torch_distributed.get_process_index_and_count(
      group
  )
  filename = directory / f"process_{process_index}-of-{process_count}.json"
  if not await asyncio.to_thread(filename.exists):
    raise ValueError(f"File {filename} does not exist.")
  state = await asyncio.to_thread(filename.read_text)
  set_state_fn(state)
