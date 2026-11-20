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

import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing as mp
from grain._src.python.dataset import dataset
from grain._src.python.experimental.torch import checkpoint as torch_checkpoint
import grain.experimental.torch as grain_torch
import portpicker

dist: Any = None
dcp: Any = None
torch_mp: Any = None
try:
  import torch.distributed as dist  # pylint: disable=g-import-not-at-top
  import torch.distributed.checkpoint as dcp  # pylint: disable=g-import-not-at-top
  import torch.multiprocessing as torch_mp  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


class _BytesIterator:

  def __init__(self, payload: bytes):
    self.payload = payload

  def get_state(self) -> bytes:
    return self.payload

  def set_state(self, state: bytes) -> None:
    self.payload = state


def _init_process_group(rank: int, world_size: int, port: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _check_process_index_and_count(
    expected: tuple[int, int], group: Any = None
) -> None:
  actual = grain_torch.get_process_index_and_count(group)
  if actual != expected:
    raise AssertionError(
        f"Unexpected rank and count: {actual!r}, expected {expected!r}"
    )


def _check_dcp_round_trip(ckpt_dir: str, group: Any = None) -> None:
  """Saves and restores this process's iterator state of a 2-shard dataset."""
  process_index, process_count = grain_torch.get_process_index_and_count(group)
  ds = dataset.MapDataset.range(20)[
      process_index::process_count
  ].to_iter_dataset()
  it = iter(ds)
  # Advance shard 0 and shard 1 by different step counts to verify DCP does not
  # deduplicate and overwrite one shard's state with the other's.
  steps = 2 if process_index == 0 else 5
  for _ in range(steps):
    next(it)
  dcp.save(
      {"data": grain_torch.IteratorCheckpointWrapper(it, group=group)},
      checkpoint_id=ckpt_dir,
  )
  dist.barrier()

  restored_it = iter(ds)
  dcp.load(
      {"data": grain_torch.IteratorCheckpointWrapper(restored_it, group=group)},
      checkpoint_id=ckpt_dir,
  )
  expected_next = 4 if process_index == 0 else 11
  actual_next = next(restored_it)
  if actual_next != expected_next:
    raise AssertionError(
        f"Shard {process_index}: expected {expected_next}, got {actual_next}"
    )


def _dcp_two_rank_worker(rank: int, port: int, ckpt_dir: str) -> None:
  _init_process_group(rank, world_size=2, port=port)
  try:
    _check_process_index_and_count(expected=(rank, 2))
    _check_dcp_round_trip(ckpt_dir)
  finally:
    dist.destroy_process_group()


def _dcp_dp_subgroup_worker(rank: int, port: int, ckpt_dir: str) -> None:
  # 2 data-parallel (DP) x 2 tensor-parallel (TP) ranks, where
  # `rank == 2 * dp_index + tp_index`. TP peers share a DP index and consume
  # the same shard.
  _init_process_group(rank, world_size=4, port=port)
  try:
    # Every rank must create every group, in the same order.
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    dp_index, tp_index = divmod(rank, 2)
    _check_process_index_and_count(
        expected=(dp_index, 2), group=dp_groups[tp_index]
    )
    try:
      grain_torch.get_process_index_and_count(dp_groups[1 - tp_index])
    except ValueError:
      pass
    else:
      raise AssertionError(
          f"Rank {rank}: expected `ValueError` for a group it is not in."
      )
    _check_dcp_round_trip(ckpt_dir, group=dp_groups[tp_index])
  finally:
    dist.destroy_process_group()


class TorchCheckpointTest(parameterized.TestCase):

  def test_exported_in_grain_experimental_torch(self):
    self.assertIs(
        grain_torch.IteratorCheckpointWrapper,
        torch_checkpoint.IteratorCheckpointWrapper,
    )

  def test_iterator_state_round_trip(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    it = iter(ds)
    self.assertEqual([next(it) for _ in range(4)], [0, 1, 2, 3])
    state_dict = torch_checkpoint.IteratorCheckpointWrapper(it).state_dict()
    self.assertEqual(state_dict["version"], 1)
    self.assertEqual(state_dict["count"], 1)
    self.assertIsInstance(state_dict["rank_0"], str)

    restored_it = iter(ds)
    torch_checkpoint.IteratorCheckpointWrapper(restored_it).load_state_dict(
        state_dict
    )
    self.assertEqual(list(restored_it), [4, 5, 6, 7, 8, 9])

  @parameterized.named_parameters(
      dict(
          testcase_name="unsupported_version",
          state_dict={"version": 99, "count": 1, "rank_0": "{}"},
          error="version",
      ),
      dict(
          testcase_name="count_mismatch",
          state_dict={"version": 1, "count": 4, "rank_0": "{}"},
          error="count mismatch",
      ),
      dict(
          testcase_name="missing_rank",
          state_dict={"version": 1, "count": 1},
          error="Missing checkpoint state",
      ),
  )
  def test_load_state_dict_rejects_invalid_state(
      self, state_dict: dict[str, Any], error: str
  ):
    it = iter(dataset.MapDataset.range(10).to_iter_dataset())
    with self.assertRaisesRegex(ValueError, error):
      torch_checkpoint.IteratorCheckpointWrapper(it).load_state_dict(state_dict)

  def test_bytes_iterator_round_trip(self):
    it = _BytesIterator(b"initial")
    saved = torch_checkpoint.IteratorCheckpointWrapper(it).state_dict()
    target = _BytesIterator(b"other")
    torch_checkpoint.IteratorCheckpointWrapper(target).load_state_dict(saved)
    self.assertEqual(target.payload, b"initial")

  @absltest.skipIf(
      dcp is None or torch_mp is None,
      "torch.distributed.checkpoint is not available.",
  )
  def test_dcp_two_ranks_preserve_distinct_states(self):
    ckpt_dir = self.create_tempdir().full_path
    port = portpicker.pick_unused_port()
    torch_mp.spawn(
        _dcp_two_rank_worker,
        args=(port, ckpt_dir),
        nprocs=2,
        join=True,
    )

  @absltest.skipIf(
      dcp is None or torch_mp is None,
      "torch.distributed.checkpoint is not available.",
  )
  def test_dcp_dp_subgroups_preserve_distinct_states(self):
    ckpt_dir = self.create_tempdir().full_path
    port = portpicker.pick_unused_port()
    torch_mp.spawn(
        _dcp_dp_subgroup_worker,
        args=(port, ckpt_dir),
        nprocs=4,
        join=True,
    )


if __name__ == "__main__":
  absltest.main()
