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

import dataclasses
import importlib
import operator
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import grain
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.experimental.torch import random_map as torch_random_map
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np

torch: Any = None
torch_random: Any = None
try:
  import torch  # pylint: disable=g-import-not-at-top
  from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

_TorchStatelessRandomMap = torch_random_map.TorchStatelessRandomMap


def _draw(key: Any, n: int = 4) -> list[int]:
  return torch_random.randint(
      key, (n,), low=0, high=2**27, dtype=torch.int64
  ).tolist()


def _with_random_ints(element: int, key: Any) -> tuple[int, list[int]]:
  return element, _draw(key, n=2)


def _make_dataset(
    seed: int = 42, num_elements: int = 8
) -> dataset.MapDataset[tuple[int, list[int]]]:
  return dataset.MapDataset.range(num_elements).stateless_random_map(
      _TorchStatelessRandomMap(_with_random_ints), seed=seed
  )


@dataclasses.dataclass(frozen=True)
class TorchRandomMapWithTransform:

  def __call__(self, element: int, key: Any) -> float:
    delta = 0.1
    return (
        element + torch_random.uniform(key, (), low=-delta, high=delta).item()
    )


@dataclasses.dataclass(frozen=True)
class TorchRandomMapWithDeterminismTransform:

  def __call__(self, element: int, key: Any) -> int:
    return (
        element
        + torch_random.randint(
            key, (), low=0, high=10, dtype=torch.int64
        ).item()
    )


class TorchRandomMapWithElementSpecInference:

  def __call__(self, element: int, key: Any) -> dict[str, dict[str, int]]:
    return {
        "nested": {
            "record": (
                element
                + torch_random.randint(
                    key, (), low=0, high=10, dtype=torch.int64
                ).item()
            )
        },
    }

  def output_spec(self, input_spec: Any) -> Any:
    assert input_spec == base.ShapeDtypeStruct(
        shape=(), dtype=np.int64
    ), input_spec
    return {
        "nested": {"record": base.ShapeDtypeStruct(shape=(), dtype=np.int64)},
    }


@absltest.skipIf(
    torch_random is None, "PyTorch stateless RNG is not available."
)
class TorchStatelessRandomMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)

  def test_map_size(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=0
    )
    self.assertLen(ds, len(self.range_ds))

  def test_map_picklable(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=0
    )
    restored_before = cloudpickle.loads(cloudpickle.dumps(ds))  # pylint: disable=g-unsafe-pickle-load
    self.assertLen(restored_before, len(self.range_ds))
    _ = ds[0]
    self.assertIn("_key_for_index", ds.__dict__)
    restored_after = cloudpickle.loads(cloudpickle.dumps(ds))  # pylint: disable=g-unsafe-pickle-load
    self.assertNotIn("_key_for_index", restored_after.__dict__)
    self.assertEqual(list(restored_before), list(ds))
    self.assertEqual(list(restored_after), list(ds))

  def test_torch_random_map_data_with_transform(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=0
    )
    expected_data = list(range(10))
    actual_data = list(ds)
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)

  def test_torch_random_map_with_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed).stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform())
    )
    ds2 = self.range_ds.seed(seed).stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform())
    )
    np.testing.assert_equal(list(ds1), list(ds2))

  def test_torch_random_map_overrides_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed).stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform())
    )
    ds2 = self.range_ds.seed(seed).stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=seed + 1
    )
    np.testing.assert_array_compare(operator.__ne__, list(ds1), list(ds2))

  def test_torch_random_map_raises_with_no_seed(self):
    with self.assertRaisesRegex(ValueError, "requires a seed"):
      self.range_ds.stateless_random_map(
          _TorchStatelessRandomMap(TorchRandomMapWithTransform())
      )

  @parameterized.parameters(
      dict(indices=list(range(10))),
      dict(indices=[0, 4, 8]),
      dict(indices=[3, 1, 7, 2]),
      dict(indices=[5, 1, 5]),
  )
  def test_torch_random_map_data_with_get_items(self, indices):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform()),
        seed=42,
    )
    expected_data = [ds[i] for i in indices]
    actual_data = ds._getitems(indices)  # pylint: disable=protected-access
    np.testing.assert_equal(expected_data, actual_data)

  def test_torch_random_map_checkpointing(self):
    ds = _make_dataset(num_elements=10).repeat(2)
    stored_it = iter(ds)
    for _ in range(3):
      next(stored_it)
    state = stored_it.get_state()
    expected = [next(stored_it) for _ in range(10)]

    loaded_it = iter(ds)
    loaded_it.set_state(state)
    actual = [next(loaded_it) for _ in range(10)]

    self.assertEqual(actual, expected)

  def test_torch_random_map_element_spec_inference(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithElementSpecInference()),
        seed=0,
    )
    self.assertEqual(
        ds._element_spec,  # pylint: disable=protected-access
        {
            "nested": {
                "record": base.ShapeDtypeStruct(shape=(), dtype=np.int64)
            },
        },
    )

  def test_torch_random_map_element_spec_inference_raises_error(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=0
    )
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec  # pylint: disable=protected-access

  def test_cross_version_determinism(self):
    # This test validates random map determinism across different versions of
    # Grain and PyTorch given a fixed seed. Note that we do not guarantee
    # cross-version determinism: `torch.func._random` is experimental and its
    # random streams may change. Only update the values if you know what you're
    # doing.
    ds = (
        dataset.MapDataset.range(10)
        .seed(41)
        .stateless_random_map(
            _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform())
        )
    )
    self.assertEqual(list(ds), [5, 5, 3, 4, 5, 11, 9, 7, 11, 15])

  def test_transform_receives_cpu_uint64_philox_key(self):
    ds = dataset.MapDataset.range(3).stateless_random_map(
        _TorchStatelessRandomMap(lambda element, key: (element, key)),
        seed=42,
    )
    items = list(ds)
    self.assertLen(items, 3)
    op_key = torch_random.key(42)
    for index, (element, key) in enumerate(items):
      self.assertEqual(element, index)
      self.assertIsInstance(key, torch.Tensor)
      self.assertEqual(key.dtype, torch.uint64)
      self.assertEqual(key.shape, (2,))
      self.assertEqual(key.device.type, "cpu")
      self.assertTrue(torch.equal(key, torch_random.fold_in(op_key, index)))

  def test_key_is_stateless_and_splittable(self):
    def split_and_draw(element: int, key: Any) -> tuple[int, bool, bool]:
      same_draw = torch.equal(
          torch_random.normal(key, (4,)),
          torch_random.normal(key, (4,)),
      )
      k1, k2 = torch_random.split(key)
      different_split_draws = not torch.equal(
          torch_random.normal(k1, (4,)),
          torch_random.normal(k2, (4,)),
      )
      return element, same_draw, different_split_draws

    ds = dataset.MapDataset.range(4).stateless_random_map(
        _TorchStatelessRandomMap(split_and_draw), seed=42
    )
    self.assertEqual(
        list(ds),
        [(i, True, True) for i in range(4)],
    )

  def test_is_deterministic(self):
    self.assertEqual(list(_make_dataset()), list(_make_dataset()))

  def test_values_depend_on_index_and_seed(self):
    values = [draws for _, draws in _make_dataset()]
    self.assertLen({tuple(draws) for draws in values}, len(values))
    other_seed_values = [draws for _, draws in _make_dataset(seed=43)]
    self.assertNotEqual(values, other_seed_values)

  @parameterized.named_parameters(
      dict(testcase_name="zero", seed=0),
      dict(testcase_name="uint32_max", seed=2**32 - 1),
      dict(testcase_name="int64_boundary", seed=2**63),
      dict(testcase_name="uint64_max", seed=2**64 - 1),
  )
  def test_supports_full_uint64_seed_range(self, seed: int):
    ds = _make_dataset(seed=seed, num_elements=4)
    self.assertEqual(list(ds), list(_make_dataset(seed=seed, num_elements=4)))

  @parameterized.named_parameters(
      dict(testcase_name="negative", seed=-1),
      dict(testcase_name="overflow", seed=2**64),
  )
  def test_out_of_range_seed_raises(self, seed: int):
    with self.assertRaisesRegex(ValueError, r"Seed must be in \[0, 2\*\*64\)"):
      dataset.MapDataset.range(4).stateless_random_map(
          _TorchStatelessRandomMap(_with_random_ints), seed=seed
      )

  def test_random_access_matches_iteration(self):
    ds = _make_dataset()
    values = list(ds)
    for index in reversed(range(len(ds))):
      self.assertEqual(ds[index], values[index])

  def test_slice(self):
    ds = _make_dataset()
    self.assertEqual(list(ds[2:5]), [ds[2], ds[3], ds[4]])

  def test_repeat_produces_distinct_epochs(self):
    ds = _make_dataset(num_elements=4).repeat(2)
    items = list(ds)
    epoch_0 = [draws for _, draws in items[:4]]
    epoch_1 = [draws for _, draws in items[4:]]
    self.assertNotEqual(epoch_0, epoch_1)

  def test_random_map_transform_raises(self):

    class LegacyTransform(transforms.RandomMap):

      def random_map(self, element: int, rng: Any) -> int:
        del rng
        return element

    with self.assertRaisesRegex(TypeError, "expects a callable"):
      _TorchStatelessRandomMap(LegacyTransform())  # pyrefly: ignore[bad-argument-type]

    with self.assertRaisesRegex(TypeError, "expects a callable"):
      torch_random_map.as_torch_stateless_random_map(
          LegacyTransform(), "stateless_random_map"  # pyrefly: ignore[bad-argument-type]
      )

  def test_docstring_example(self):
    def add_noise(x: Any, key: Any) -> Any:
      return x + torch_random.normal(key, x.shape)

    grain_torch = grain.experimental.torch
    ds = (
        grain_torch.MapDataset.source([torch.zeros(2)] * 4)
        .stateless_random_map(add_noise, seed=42)
        .batch(2)
    )
    batch = ds[1]
    self.assertIsNotNone(batch)
    self.assertEqual(batch.shape, (2, 2))
    self.assertTrue(torch.equal(ds[1], ds[1]))
    self.assertFalse(torch.equal(ds[0], ds[1]))

    # Core Grain datasets need the explicit wrapper and produce the same
    # values.
    core_ds = grain.MapDataset.source(
        [torch.zeros(2)] * 4
    ).stateless_random_map(
        grain_torch.TorchStatelessRandomMap(add_noise), seed=42
    )
    self.assertTrue(torch.equal(torch.stack([core_ds[2], core_ds[3]]), ds[1]))

  def test_none_elements_are_not_mapped(self):
    ds = (
        dataset.MapDataset.range(6)
        .filter(lambda x: x % 2 == 0)
        .stateless_random_map(
            _TorchStatelessRandomMap(lambda element, key: element + 10),
            seed=1,
        )
    )
    self.assertIsNone(ds[1])
    self.assertEqual(list(ds), [10, 12, 14])

  def test_global_rng_is_not_used_or_modified(self):
    torch.manual_seed(0)
    state = torch.get_rng_state()
    values = list(_make_dataset())
    self.assertTrue(torch.equal(torch.get_rng_state(), state))
    torch.manual_seed(123)
    self.assertEqual(list(_make_dataset()), values)

  def test_multithreaded_prefetch_is_deterministic(self):
    ds = _make_dataset(num_elements=64)
    read_options = options.ReadOptions(num_threads=8, prefetch_buffer_size=16)
    self.assertEqual(list(ds.to_iter_dataset(read_options)), list(ds))

  def test_multiprocess_prefetch_is_deterministic(self):
    ds = _make_dataset(num_elements=16)
    mp_options = options.MultiprocessingOptions(num_workers=2)
    self.assertEqual(
        list(ds.to_iter_dataset().mp_prefetch(mp_options)), list(ds)
    )

  def test_str(self):
    ds = self.range_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithTransform()), seed=0
    )
    self.assertStartsWith(str(ds), "StatelessRandomMapMapDataset(transform=")
    self.assertIn("TorchRandomMapWithTransform", str(ds))

  def test_exported_exclusively_under_grain_experimental_torch(self):
    # `import grain.experimental.torch`, `from grain.experimental import torch`
    # and attribute access must all resolve to the same module.
    grain_torch = importlib.import_module("grain.experimental.torch")
    self.assertIs(grain.experimental.torch, grain_torch)
    self.assertIs(
        grain_torch.TorchStatelessRandomMap,
        _TorchStatelessRandomMap,
    )
    self.assertFalse(hasattr(grain.experimental, "TorchRandomMapMapDataset"))
    self.assertFalse(hasattr(grain.experimental, "TorchRandomMapIterDataset"))

  def test_torch_is_imported_lazily_on_element_access(self):
    with mock.patch.dict(
        "sys.modules",
        {"torch": None, "torch.func": None, "torch.func._random": None},
    ):
      ds = dataset.MapDataset.range(4).stateless_random_map(
          _TorchStatelessRandomMap(lambda x, key: x), seed=42
      )
      with self.assertRaises(ImportError):
        _ = ds[0]

  def test_subclassing_and_apply(self):
    class SubclassedAugment(_TorchStatelessRandomMap):

      def stateless_random_map(self, element: int, key: Any) -> int:
        return (
            element
            + torch_random.randint(
                key, (), low=0, high=10, dtype=torch.int64
            ).item()
        )

    wrapped = _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform())
    ds_wrapped = dataset.MapDataset.range(10).stateless_random_map(
        wrapped, seed=41
    )
    ds_subclassed = dataset.MapDataset.range(10).stateless_random_map(
        SubclassedAugment(), seed=41
    )
    ds_applied = (
        dataset.MapDataset.range(10).seed(41).apply(SubclassedAugment())
    )
    self.assertEqual(list(ds_wrapped), list(ds_subclassed))
    self.assertEqual(
        list(ds_applied),
        [5, 5, 3, 4, 5, 11, 9, 7, 11, 15],
    )


@absltest.skipIf(
    torch_random is None, "PyTorch stateless RNG is not available."
)
class TorchStatelessRandomMapIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(0, 10).to_iter_dataset()

  def test_deterministic_and_matches_map_dataset(self):
    ds_iter = self.range_iter_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform()),
        seed=42,
    )
    ds_map = dataset.MapDataset.range(0, 10).stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform()),
        seed=42,
    )
    self.assertEqual(list(ds_iter), list(ds_map))
    self.assertEqual(list(ds_iter), [3, 9, 4, 3, 12, 13, 12, 13, 10, 15])

  @parameterized.parameters(0, 1, 42)
  def test_checkpointing(self, seed: int):
    ds = self.range_iter_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform()),
        seed=seed,
    )
    assert_equal_output_after_checkpoint(ds)

  def test_multiprocess_prefetch_shifts_worker_rng_indices(self):
    ds = (
        dataset.MapDataset.range(16)
        .to_iter_dataset()
        .stateless_random_map(
            _TorchStatelessRandomMap(_with_random_ints), seed=7
        )
        .mp_prefetch(options.MultiprocessingOptions(num_workers=2))
    )
    items = list(ds)
    draws = [tuple(vals) for _, vals in items]
    self.assertLen(set(draws), 16)
    self.assertEqual(list(ds), items)

  def test_get_and_set_next_index(self):
    ds = self.range_iter_ds.stateless_random_map(
        _TorchStatelessRandomMap(TorchRandomMapWithDeterminismTransform()),
        seed=0,
    )
    expected = list(ds)
    ds_iter = ds.__iter__()
    for i in reversed(range(10)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      self.assertEqual(next(ds_iter), expected[i])


if __name__ == "__main__":
  absltest.main()
