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

import asyncio
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
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


class _ImmediatePath:

  def __init__(self, path: epath.Path):
    self._path = path

  async def await_creation(self) -> epath.Path:
    return self._path


class _BytesIterator:

  def __init__(self, payload: bytes):
    self.payload = payload

  def get_state(self) -> bytes:
    return self.payload

  def set_state(self, state: bytes) -> None:
    self.payload = state


def _dcp_two_rank_worker(rank: int, port: int, ckpt_dir: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  dist.init_process_group("gloo", rank=rank, world_size=2)
  try:
    process_index, process_count = grain_torch.get_process_index_and_count()
    if (process_index, process_count) != (rank, 2):
      raise AssertionError(
          f"Unexpected rank and count: {(process_index, process_count)!r}"
      )
    ds = dataset.MapDataset.range(20)[
        process_index::process_count
    ].to_iter_dataset()
    it = iter(ds)
    # Advance rank 0 and rank 1 by different step counts to verify DCP does not
    # deduplicate and overwrite rank 1 with rank 0's state.
    steps = 2 if rank == 0 else 5
    for _ in range(steps):
      next(it)
    dcp.save({"data": grain_torch.IteratorState(it)}, checkpoint_id=ckpt_dir)
    dist.barrier()

    restored_it = iter(ds)
    dcp.load(
        {"data": grain_torch.IteratorState(restored_it)},
        checkpoint_id=ckpt_dir,
    )
    expected_next = 4 if rank == 0 else 11
    actual_next = next(restored_it)
    if actual_next != expected_next:
      raise AssertionError(
          f"Rank {rank}: expected {expected_next}, got {actual_next}"
      )
  finally:
    dist.destroy_process_group()


class TorchCheckpointTest(parameterized.TestCase):

  def test_exported_in_grain_experimental_torch(self):
    self.assertIs(grain_torch.IteratorState, torch_checkpoint.IteratorState)

  def test_iterator_state_round_trip_and_validation(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    it = iter(ds)
    self.assertEqual([next(it) for _ in range(4)], [0, 1, 2, 3])
    state_dict = torch_checkpoint.IteratorState(it).state_dict()
    self.assertEqual(state_dict["version"], 1)
    self.assertEqual(state_dict["count"], 1)
    self.assertIsInstance(state_dict["rank_0"], str)

    restored_it = iter(ds)
    torch_checkpoint.IteratorState(restored_it).load_state_dict(state_dict)
    self.assertEqual(list(restored_it), [4, 5, 6, 7, 8, 9])

    with self.assertRaisesRegex(ValueError, "version"):
      torch_checkpoint.IteratorState(restored_it).load_state_dict(
          {"version": 99, "count": 1, "rank_0": "{}"}
      )
    with self.assertRaisesRegex(ValueError, "count mismatch"):
      torch_checkpoint.IteratorState(restored_it).load_state_dict(
          {"version": 1, "count": 4, "rank_0": "{}"}
      )
    with self.assertRaisesRegex(ValueError, "Missing checkpoint state"):
      torch_checkpoint.IteratorState(restored_it).load_state_dict(
          {"version": 1, "count": 1}
      )

  def test_bytes_iterator_round_trip(self):
    it = _BytesIterator(b"initial")
    saved = torch_checkpoint.IteratorState(it).state_dict()
    target = _BytesIterator(b"other")
    torch_checkpoint.IteratorState(target).load_state_dict(saved)
    self.assertEqual(target.payload, b"initial")

  def test_background_save_and_load_uses_torch_rank(self):
    ckpt_dir = epath.Path(self.create_tempdir().full_path)
    with mock.patch.dict(
        os.environ, {"RANK": "1", "WORLD_SIZE": "4"}, clear=False
    ):
      asyncio.run(
          torch_checkpoint.background_save(
              _ImmediatePath(ckpt_dir), '{"next_index": 7}'
          )
      )
      expected_file = ckpt_dir / "process_1-of-4.json"
      self.assertTrue(expected_file.exists())
      loaded: list[str] = []
      asyncio.run(torch_checkpoint.background_load(ckpt_dir, loaded.append))
      self.assertEqual(loaded, ['{"next_index": 7}'])

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


if __name__ == "__main__":
  absltest.main()
