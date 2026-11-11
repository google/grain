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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grain
import multiprocessing as mp
from grain._src.python import options as grain_options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.experimental.torch import options as torch_options
from grain._src.python.experimental.torch import prefetch as torch_prefetch
from grain._src.python.ipc import shared_memory_array
import numpy as np

torch_lib: Any = None
try:
  import torch as torch_lib  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

tv_tensors: Any = None
try:
  from torchvision import tv_tensors  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


if torch_lib is not None:

  class _TaggedTensor(torch_lib.Tensor):

    @staticmethod
    def __new__(cls, data: Any, tag: str = "default"):
      obj = torch_lib.Tensor._make_subclass(cls, data, require_grad=False)
      obj.tag = tag
      return obj


def _worker_num_threads(_: int) -> int:
  return torch_lib.get_num_threads()


def _set_four_threads(worker_index: int, worker_count: int) -> None:
  del worker_index, worker_count
  torch_lib.set_num_threads(4)


def _make_dtype_tensor(dtype_name: str, index: int) -> Any:
  dtype = getattr(torch_lib, dtype_name)
  if dtype == torch_lib.bool:
    return torch_lib.tensor([index % 2 == 0, index % 2 == 1], dtype=dtype)
  if dtype.is_complex:
    return torch_lib.tensor(
        [complex(float(index), 1.5), complex(-2.0, float(index))], dtype=dtype
    )
  if dtype.is_floating_point:
    return torch_lib.tensor(
        [float(index) + 0.25, float(index) - 0.5], dtype=dtype
    )
  return torch_lib.tensor([index, index + 3], dtype=dtype)


@absltest.skipIf(torch_lib is None, "PyTorch is not available.")
class TorchPrefetchThreadCapTest(parameterized.TestCase):

  def test_default_caps_workers_to_one_thread_without_mutating_parent(self):
    parent_before = torch_lib.get_num_threads()
    parent_omp = os.environ.get("OMP_NUM_THREADS")
    ds = (
        dataset.MapDataset.range(4)
        .map(_worker_num_threads)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            grain_options.MultiprocessingOptions(num_workers=2),
        )
    )
    self.assertEqual(list(ds), [1, 1, 1, 1])
    self.assertEqual(torch_lib.get_num_threads(), parent_before)
    self.assertEqual(os.environ.get("OMP_NUM_THREADS"), parent_omp)

  def test_custom_intra_op_threads_and_mkl_override(self):
    with mock.patch.dict(os.environ, {"MKL_NUM_THREADS": "8"}, clear=False):
      ds = (
          dataset.MapDataset.range(4)
          .map(_worker_num_threads)
          .to_iter_dataset()
          .pipe(
              torch_prefetch.mp_prefetch,
              torch_options.MultiprocessingOptions(
                  num_workers=2, worker_intra_op_threads=2
              ),
          )
      )
      self.assertEqual(list(ds), [2, 2, 2, 2])

  def test_worker_init_sets_torch_threads_before_user_init_fn(self):
    calls = []
    with (
        mock.patch.dict(os.environ, clear=False),
        mock.patch.object(
            torch_lib,
            "set_num_threads",
            side_effect=lambda n: calls.append(("set_num_threads", n)),
        ),
    ):
      torch_prefetch._torch_worker_init(  # pylint: disable=protected-access
          0,
          2,
          intra_op_threads=3,
          user_init_fn=lambda index, count: calls.append(("user", index)),
      )
      self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
      self.assertEqual(os.environ["MKL_NUM_THREADS"], "3")
    self.assertEqual(calls, [("set_num_threads", 3), ("user", 0)])

  def test_user_worker_init_fn_sets_threads_when_cap_is_disabled(self):
    ds = (
        dataset.MapDataset.range(4)
        .map(_worker_num_threads)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(
                num_workers=2, worker_intra_op_threads=None
            ),
            worker_init_fn=_set_four_threads,
        )
    )
    self.assertEqual(list(ds), [4, 4, 4, 4])

  def test_none_intra_op_threads_leaves_env_uncapped(self):
    with mock.patch.dict(
        os.environ,
        {"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"},
        clear=False,
    ):
      ds = (
          dataset.MapDataset.range(2)
          .map(_worker_num_threads)
          .to_iter_dataset()
          .pipe(
              torch_prefetch.mp_prefetch,
              torch_options.MultiprocessingOptions(
                  num_workers=1, worker_intra_op_threads=None
              ),
          )
      )
      self.assertEqual(list(ds), [3, 3])

  def test_invalid_worker_intra_op_threads_raises(self):
    with self.assertRaisesRegex(ValueError, "worker_intra_op_threads"):
      torch_options.MultiprocessingOptions(
          num_workers=2, worker_intra_op_threads=0
      )


@absltest.skipIf(torch_lib is None, "PyTorch is not available.")
class TorchPrefetchIpcTest(parameterized.TestCase):

  @parameterized.parameters(
      "float32",
      "float16",
      "bfloat16",
      "int8",
      "int32",
      "int64",
      "uint8",
      "bool",
      "complex64",
  )
  def test_round_trip_dtypes(self, dtype_name: str):
    ds = (
        dataset.MapDataset.range(4)
        .map(lambda i: _make_dtype_tensor(dtype_name, i))
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(num_workers=2),
        )
    )
    results = list(ds)
    self.assertLen(results, 4)
    for i, actual in enumerate(results):
      expected = _make_dtype_tensor(dtype_name, i)
      self.assertEqual(actual.dtype, expected.dtype)
      self.assertEqual(actual.shape, expected.shape)
      self.assertTrue(torch_lib.equal(actual, expected))

  def test_special_shapes_non_contiguous_neg_and_conj(self):
    def make_special(i: int) -> dict[str, Any]:
      mat = torch_lib.arange(6, dtype=torch_lib.float32).reshape(2, 3) + i
      comp = torch_lib.tensor(
          [complex(float(i), 2.0)], dtype=torch_lib.complex64
      )
      return {
          "scalar": torch_lib.tensor(i, dtype=torch_lib.int64),
          "empty": torch_lib.zeros((0, 3), dtype=torch_lib.bfloat16),
          "transposed": mat.t(),
          "neg": -mat,
          "conj": comp.conj(),
      }

    ds = (
        dataset.MapDataset.range(4)
        .map(make_special)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(num_workers=2),
        )
    )
    for i, got in enumerate(ds):
      want = make_special(i)
      for key in want:
        self.assertTrue(torch_lib.equal(got[key], want[key]), msg=key)

  def test_min_shm_size_controls_inline_vs_shared_memory_backing(self):
    def make_pair(i: int) -> tuple[Any, Any]:
      small = torch_lib.full((4,), i, dtype=torch_lib.float32)  # 16 bytes
      large = torch_lib.full((1024,), i, dtype=torch_lib.float32)  # 4096 bytes
      return small, large

    iter_ds = (
        dataset.MapDataset.range(4)
        .map(make_pair)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(num_workers=2),
        )
    )
    iter_ds = dataset.WithOptionsIterDataset(
        iter_ds, base.DatasetOptions(min_shm_size=256)
    )
    small_out, large_out = next(iter(iter_ds))
    self.assertTrue(
        torch_lib.equal(
            small_out, torch_lib.zeros((4,), dtype=torch_lib.float32)
        )
    )
    self.assertTrue(
        torch_lib.equal(
            large_out, torch_lib.zeros((1024,), dtype=torch_lib.float32)
        )
    )
    # Small tensor (< 256B) is pickled inline; large tensor (>= 256B) is backed
    # by SharedMemoryArray.
    self.assertNotIsInstance(
        small_out.numpy(), shared_memory_array.SharedMemoryArray
    )
    # Verify via _decode_leaf directly that SharedMemoryArray stays zero-copy:
    shm_arr = shared_memory_array.SharedMemoryArray((8,), dtype=np.uint8)
    shm_arr[:] = 1
    decoded = torch_prefetch._decode_leaf(  # pylint: disable=protected-access
        torch_prefetch._IpcTensor(  # pylint: disable=protected-access
            data=shm_arr,
            meta=torch_prefetch._TensorMeta(  # pylint: disable=protected-access
                dtype=torch_lib.uint8,
                shape=(8,),
                is_complex=False,
                tensor_cls=torch_lib.Tensor,
                attrs=None,
            ),
        )
    )
    self.assertEqual(decoded.data_ptr(), shm_arr.ctypes.data)

  def test_tensor_subclass_and_requires_grad_fallback(self):
    def make_items(i: int) -> dict[str, Any]:
      sub = _TaggedTensor(
          torch_lib.tensor([float(i), float(i + 1)], dtype=torch_lib.bfloat16),
          tag=f"sample_{i}",
      )
      grad_t = torch_lib.tensor([float(i)], requires_grad=True)
      return {"sub": sub, "grad": grad_t, "label": f"item_{i}"}

    ds = (
        dataset.MapDataset.range(4)
        .map(make_items)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(num_workers=2),
        )
    )
    results = list(ds)
    for i, item in enumerate(results):
      self.assertIs(type(item["sub"]), _TaggedTensor)
      self.assertEqual(item["sub"].tag, f"sample_{i}")
      self.assertEqual(item["sub"].dtype, torch_lib.bfloat16)
      self.assertTrue(item["grad"].requires_grad)
      self.assertEqual(item["label"], f"item_{i}")

  @absltest.skipIf(tv_tensors is None, "torchvision is not available.")
  def test_torchvision_tv_tensors_keep_type_and_metadata(self):
    def make_items(i: int) -> dict[str, Any]:
      return {
          "image": tv_tensors.Image(
              torch_lib.full((3, 4, 4), i, dtype=torch_lib.uint8)
          ),
          "boxes": tv_tensors.BoundingBoxes(
              torch_lib.tensor([[0, 0, i + 1, i + 1]]),
              format=tv_tensors.BoundingBoxFormat.XYXY,
              canvas_size=(4, 4),
          ),
          "mask": tv_tensors.Mask(
              torch_lib.zeros((4, 4), dtype=torch_lib.bool)
          ),
      }

    # `tv_tensors` take the shared-memory path rather than torch's own pickling.
    for leaf in make_items(0).values():
      self.assertIsInstance(
          torch_prefetch._encode_leaf(leaf),  # pylint: disable=protected-access
          torch_prefetch._IpcTensor,  # pylint: disable=protected-access
      )

    ds = (
        dataset.MapDataset.range(4)
        .map(make_items)
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(num_workers=2),
        )
    )
    for i, item in enumerate(ds):
      self.assertIs(type(item["image"]), tv_tensors.Image)
      self.assertTrue(
          torch_lib.equal(
              item["image"], torch_lib.full((3, 4, 4), i, dtype=torch_lib.uint8)
          )
      )
      self.assertIs(type(item["boxes"]), tv_tensors.BoundingBoxes)
      self.assertEqual(item["boxes"].format, tv_tensors.BoundingBoxFormat.XYXY)
      self.assertEqual(item["boxes"].canvas_size, (4, 4))
      self.assertEqual(item["boxes"].tolist(), [[0, 0, i + 1, i + 1]])
      self.assertIs(type(item["mask"]), tv_tensors.Mask)

  def test_checkpoint_interoperability_with_plain_mp_prefetch(self):
    base_iter_ds = (
        dataset.MapDataset.range(12)
        .map(lambda i: torch_lib.tensor([i, i * 10]))
        .to_iter_dataset()
    )
    plain_ds = base_iter_ds.mp_prefetch(
        grain_options.MultiprocessingOptions(num_workers=2)
    )
    torch_ds = torch_prefetch.mp_prefetch(
        base_iter_ds, torch_options.MultiprocessingOptions(num_workers=2)
    )

    plain_it = iter(plain_ds)
    for _ in range(5):
      next(plain_it)
    state = plain_it.get_state()
    expected_rest = [next(plain_it).tolist() for _ in range(7)]
    plain_it.close()

    torch_it = iter(torch_ds)
    torch_it.set_state(state)
    actual_rest = [t.tolist() for t in torch_it]
    torch_it.close()
    self.assertEqual(actual_rest, expected_rest)

  def test_numpy_shared_memory_batch_output_forwarded_and_early_close(self):
    ds = (
        dataset.MapDataset.range(32)
        .map(lambda i: np.full((16,), i, dtype=np.int64))
        .to_iter_dataset()
        .batch(4)
        .pipe(
            torch_prefetch.mp_prefetch,
            torch_options.MultiprocessingOptions(
                num_workers=2, per_worker_buffer_size=4
            ),
        )
    )
    it = iter(ds)
    first_batch = next(it)
    self.assertIsInstance(first_batch, shared_memory_array.SharedMemoryArray)
    it.close()

  def test_early_break_on_tensor_pipeline_closes_cleanly(self):
    ds = (
        dataset.MapDataset.range(100)
        .map(lambda i: torch_lib.full((64,), i, dtype=torch_lib.bfloat16))
        .to_iter_dataset()
        .pipe(
            torch_prefetch.mp_prefetch,
            grain.experimental.torch.MultiprocessingOptions(
                num_workers=2, per_worker_buffer_size=4
            ),
        )
    )
    it = iter(ds)
    for step, batch in enumerate(it):
      self.assertEqual(batch.dtype, torch_lib.bfloat16)
      if step == 5:
        break
    it.close()


if __name__ == "__main__":
  absltest.main()
