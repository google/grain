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
"""Distributed rank detection for PyTorch."""

from __future__ import annotations

import os
from typing import Any


def _env_rank_and_world_size() -> tuple[int, int] | None:
  """Parses RANK and WORLD_SIZE environment variables if valid."""
  rank_str = os.environ.get("RANK")
  world_str = os.environ.get("WORLD_SIZE")
  if rank_str is None or world_str is None:
    return None
  try:
    rank = int(rank_str)
    world_size = int(world_str)
  except ValueError:
    return None
  if world_size < 1 or not 0 <= rank < world_size:
    return None
  return rank, world_size


def get_process_index_and_count(group: Any = None) -> tuple[int, int]:
  """Returns ``(process_index, process_count)`` for PyTorch distributed jobs.

  Resolution order:

  1. If ``torch.distributed`` is available and initialized, queries
     ``dist.get_rank(group)`` and ``dist.get_world_size(group)``. Raises
     ``ValueError`` if the current process is not a member of ``group``.
  2. If ``group`` was explicitly provided when ``torch.distributed`` is not
     initialized, raises ``RuntimeError``.
  3. Falls back to valid ``RANK`` and ``WORLD_SIZE`` environment variables
     (for ``torchrun`` before ``init_process_group``).
  4. Otherwise returns ``(0, 1)``.

  Never imports JAX, avoiding accidental PJRT initialization in PyTorch jobs.

  Example:
    Sharding a dataset across ``torch.distributed`` ranks with a slice::

      rank, world_size = grain.experimental.torch.get_process_index_and_count()
      ds = ds.shuffle().repeat(num_epochs)[rank::world_size]

  Args:
    group: Optional ``torch.distributed.ProcessGroup`` (e.g., DP group under
      2D/3D parallelism). Defaults to the default process group.

  Returns:
    Tuple ``(process_index, process_count)``.

  Raises:
    ValueError: If the current process is not a member of ``group``.
    RuntimeError: If ``group`` is specified when ``torch.distributed`` is not
      initialized.
  """
  dist = None
  try:
    import torch.distributed as dist  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
  except ImportError:
    pass

  if dist is not None and dist.is_available() and dist.is_initialized():
    rank = dist.get_rank(group)
    if rank < 0:
      raise ValueError(
          "Current process is not a member of the given process group."
      )
    return rank, dist.get_world_size(group)

  if group is not None:
    raise RuntimeError(
        "`torch.distributed` must be initialized when `group` is specified."
    )

  env_result = _env_rank_and_world_size()
  if env_result is not None:
    return env_result

  return 0, 1
