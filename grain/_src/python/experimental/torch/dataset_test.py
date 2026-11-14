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
import cloudpickle
from etils import epath
import grain
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
import grain.experimental.torch as grain_torch
import numpy as np

torch_lib: Any = None
torch_random: Any = None
dcp: Any = None
try:
  import torch as torch_lib  # pylint: disable=g-import-not-at-top
  import torch.distributed.checkpoint as dcp  # pylint: disable=g-import-not-at-top
  from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


class _ImmediatePath:

  def __init__(self, path: epath.Path):
    self._path = path

  async def await_creation(self) -> epath.Path:
    return self._path


class _AddOneMap(transforms.Map):

  def map(self, element: int) -> int:
    return element + 1


def _augment_bf16(sample: tuple[Any, int], key: Any) -> tuple[Any, int, int]:
  tensor, label = sample
  noise = torch_random.uniform(key, tensor.shape).to(torch_lib.bfloat16)
  return tensor + noise, label, torch_lib.get_num_threads()


@absltest.skipIf(
    torch_lib is None or torch_random is None,
    "PyTorch with stateless RNG is not available.",
)
class TorchDatasetEndToEndTest(parameterized.TestCase):

  def test_default_seed_depth_matches_core_grain_exactly(self):
    core_ds = (
        grain.MapDataset.range(12)
        .seed(42)
        .shuffle()
        .slice(slice(0, 10))
        .random_map(lambda x, rng: x + int(rng.integers(0, 100)))
    )
    torch_ds = (
        grain_torch.MapDataset.range(12)
        .seed(42)
        .shuffle()
        .slice(slice(0, 10))
        .random_map(lambda x, rng: x + int(rng.integers(0, 100)))
    )
    self.assertIsInstance(torch_ds, grain.MapDataset)
    self.assertIsInstance(torch_ds, grain_torch.MapDataset)
    self.assertEqual(list(torch_ds), list(core_ds))

  def test_batch_defaults_to_torch_collate_and_supports_element_spec(self):
    source = [
        (torch_lib.full((3,), float(i), dtype=torch_lib.bfloat16), i)
        for i in range(5)
    ]
    batched = grain_torch.MapDataset.source(source).batch(
        2, drop_remainder=True
    )
    self.assertLen(batched, 2)
    first_batch = batched[0]
    assert first_batch is not None
    imgs, labels = first_batch
    self.assertIsInstance(imgs, torch_lib.Tensor)
    self.assertEqual(imgs.dtype, torch_lib.bfloat16)
    self.assertEqual(imgs.shape, (2, 3))
    self.assertIsInstance(labels, torch_lib.Tensor)
    self.assertEqual(labels.tolist(), [0, 1])

    spec_ds = grain_torch.MapDataset.range(6).batch(2, drop_remainder=True)
    self.assertEqual(
        grain.experimental.get_element_spec(spec_ds),
        base.ShapeDtypeStruct(shape=(2,), dtype=np.int64),
    )

    unpickled = cloudpickle.loads(cloudpickle.dumps(batched))  # pylint: disable=g-unsafe-pickle-load
    imgs2, labels2 = unpickled[0]
    self.assertTrue(torch_lib.equal(imgs2, imgs))
    self.assertTrue(torch_lib.equal(labels2, labels))

  def test_end_to_end_pipeline_with_mp_prefetch_and_dcp_checkpoint(self):
    source = [
        (torch_lib.zeros((8,), dtype=torch_lib.bfloat16), i) for i in range(16)
    ]
    rank, world_size = grain_torch.get_process_index_and_count()
    ds = (
        grain_torch.MapDataset.source(source)
        .seed(42)
        .shuffle()
        .repeat(2)[rank::world_size]
        .stateless_random_map(_augment_bf16)
        .batch(4, drop_remainder=True)
        .to_iter_dataset()
        .with_options(grain_torch.DatasetOptions(min_shm_size=64))
        .mp_prefetch(grain_torch.MultiprocessingOptions(num_workers=2))
    )
    self.assertIsInstance(ds, grain_torch.IterDataset)

    it = iter(ds)
    self.assertIsInstance(it, grain_torch.DatasetIterator)
    first_imgs, first_labels, worker_threads = next(it)
    self.assertEqual(first_imgs.dtype, torch_lib.bfloat16)
    self.assertEqual(first_imgs.shape, (4, 8))
    self.assertEqual(first_labels.shape, (4,))
    # Default worker_intra_op_threads=1 caps every worker process to 1 thread.
    self.assertEqual(worker_threads.tolist(), [1, 1, 1, 1])

    next(it)
    if dcp is not None:
      ckpt_dir = self.create_tempdir().full_path
      dcp.save({"data": it}, checkpoint_id=ckpt_dir)
      expected_remaining = [
          (imgs.clone(), labels.clone()) for imgs, labels, _ in it
      ]
      it.close()

      restored_it = iter(ds)
      dcp.load({"data": restored_it}, checkpoint_id=ckpt_dir)
      actual_remaining = [
          (imgs.clone(), labels.clone()) for imgs, labels, _ in restored_it
      ]
      restored_it.close()
      self.assertLen(actual_remaining, len(expected_remaining))
      for (got_img, got_lbl), (want_img, want_lbl) in zip(
          actual_remaining, expected_remaining
      ):
        self.assertTrue(torch_lib.equal(got_img, want_img))
        self.assertTrue(torch_lib.equal(got_lbl, want_lbl))
    else:
      it.close()

  def test_apply_dispatches_stateless_random_map_and_torch_batch(self):
    class AddTorchNoise(grain_torch.TorchStatelessRandomMap):

      def stateless_random_map(self, element: int, key: Any) -> Any:
        return torch_lib.tensor(float(element)) + torch_random.uniform(key, ())

    map_ds = (
        grain_torch.MapDataset.range(4)
        .seed(7)
        .apply([_AddOneMap(), AddTorchNoise(), transforms.Batch(batch_size=2)])
    )
    self.assertIsInstance(map_ds, grain_torch.MapDataset)
    batch0 = map_ds[0]
    assert batch0 is not None
    self.assertIsInstance(batch0, torch_lib.Tensor)
    self.assertEqual(batch0.shape, (2,))

    iter_ds = (
        grain_torch.MapDataset.range(4)
        .to_iter_dataset()
        .seed(7)
        .apply([_AddOneMap(), AddTorchNoise(), transforms.Batch(batch_size=2)])
    )
    self.assertIsInstance(iter_ds, grain_torch.IterDataset)
    iter_batches = list(iter_ds)
    self.assertLen(iter_batches, 2)
    self.assertIsInstance(iter_batches[0], torch_lib.Tensor)

  def test_docstring_examples(self):
    # `MapDataset`.
    ds = (
        grain_torch.MapDataset.source([torch_lib.zeros(3)] * 8)
        .shuffle(seed=42)
        .batch(4)
    )
    first_batch = ds[0]
    assert first_batch is not None
    self.assertEqual(first_batch.shape, torch_lib.Size([4, 3]))

    # `batch`.
    for ds in (
        grain_torch.MapDataset.range(5),
        grain_torch.MapDataset.range(5).to_iter_dataset(),
    ):
      batches = list(ds.batch(batch_size=2))
      self.assertTrue(all(isinstance(b, torch_lib.Tensor) for b in batches))
      self.assertEqual([b.tolist() for b in batches], [[0, 1], [2, 3], [4]])

    # `from_dataset`.
    ds = grain_torch.MapDataset.from_dataset(grain.MapDataset.range(4))
    self.assertEqual([b.tolist() for b in ds.batch(2)], [[0, 1], [2, 3]])
    ds = grain_torch.IterDataset.from_dataset(
        grain.MapDataset.range(4).to_iter_dataset()
    )
    self.assertEqual([b.tolist() for b in ds.batch(2)], [[0, 1], [2, 3]])

    # `stateless_random_map`.
    def add_noise(x, key):
      return x + torch_random.normal(key, x.shape)

    ds = grain_torch.MapDataset.source([torch_lib.zeros(2)] * 4)
    map_ds = ds.seed(42).stateless_random_map(add_noise)
    iter_ds = ds.to_iter_dataset().seed(42).stateless_random_map(add_noise)
    for ds in (map_ds, iter_ds):
      first, second = list(ds), list(ds)
      self.assertLen(first, 4)
      for a, b in zip(first, second):
        self.assertTrue(torch_lib.equal(a, b))
      self.assertFalse(torch_lib.equal(first[0], first[1]))

    # `mp_prefetch`.
    ds = grain_torch.MapDataset.source([torch_lib.zeros(1024)] * 8).batch(4)
    ds = ds.to_iter_dataset().mp_prefetch(
        grain_torch.MultiprocessingOptions(num_workers=2)
    )
    it = iter(ds)
    self.assertEqual(next(it).shape, torch_lib.Size([4, 1024]))
    it.close()

  def test_mix_concatenate_from_dataset_and_slicing(self):
    d1 = grain_torch.MapDataset.from_dataset(dataset.MapDataset.range(0, 4))
    d2 = grain_torch.MapDataset.range(10, 14)
    mixed = grain_torch.MapDataset.mix([d1, d2])
    concat = grain_torch.MapDataset.concatenate([d1, d2])[1::2]
    self.assertIsInstance(mixed, grain_torch.MapDataset)
    self.assertIsInstance(concat, grain_torch.MapDataset)
    self.assertEqual(list(concat), [1, 3, 11, 13])

    iter_mixed = grain_torch.IterDataset.mix(
        {"a": d1.to_iter_dataset(), "b": d2.to_iter_dataset()}
    ).batch(2)
    batches = list(iter_mixed)
    self.assertTrue(all(isinstance(b, torch_lib.Tensor) for b in batches))

  def test_iter_dataset_stateless_random_map_and_async_rank_save_load(self):
    iter_ds = (
        grain_torch.MapDataset.range(8)
        .to_iter_dataset()
        .stateless_random_map(
            lambda x, key: x
            + int(
                torch_random.randint(
                    key, (), low=0, high=10, dtype=torch_lib.int64
                ).item()
            ),
            seed=11,
        )
    )
    it = iter(iter_ds)
    head = [next(it) for _ in range(3)]
    self.assertLen(head, 3)

    ckpt_dir = epath.Path(self.create_tempdir().full_path)

    async def _save_and_restore() -> list[int]:
      save_coro = await it.save(_ImmediatePath(ckpt_dir))
      await save_coro
      restored = iter(iter_ds)
      load_coro = await restored.load(ckpt_dir)
      await load_coro
      return list(restored)

    with mock.patch.dict(
        os.environ, {"RANK": "1", "WORLD_SIZE": "4"}, clear=False
    ):
      restored_tail = asyncio.run(_save_and_restore())
      self.assertTrue((ckpt_dir / "process_1-of-4.json").exists())
      self.assertEqual(restored_tail, list(it))

  def test_constructing_torch_pipeline_does_not_eagerly_import_torch(self):
    with mock.patch.dict(
        "sys.modules",
        {
            "torch": None,
            "torch.func": None,
            "torch.func._random": None,
            "torch.utils": None,
            "torch.utils.data": None,
        },
    ):
      ds = (
          grain_torch.MapDataset.range(4)
          .seed(1)
          .stateless_random_map(lambda x, key: x)
          .batch(2)
      )
      self.assertLen(ds, 2)
      with self.assertRaises(ImportError):
        _ = ds[0]


if __name__ == "__main__":
  absltest.main()
