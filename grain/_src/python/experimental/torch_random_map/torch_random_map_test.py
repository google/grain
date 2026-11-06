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
"""Tests for the experimental stateless PyTorch random map."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import grain
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.experimental.torch_random_map import torch_random_map

torch: Any = None
torch_random: Any = None
try:
  import torch  # pylint: disable=g-import-not-at-top
  from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

_TorchRandomMapMapDataset = torch_random_map.TorchRandomMapMapDataset


def _draw(key: Any, n: int = 4) -> list[int]:
  return torch_random.randint(
      key, (n,), low=0, high=2**27, dtype=torch.int64
  ).tolist()


def _with_random_ints(element: int, key: Any) -> tuple[int, list[int]]:
  return element, _draw(key, n=2)


def _make_dataset(
    seed: int = 42, num_elements: int = 8
) -> dataset.MapDataset[tuple[int, list[int]]]:
  return dataset.MapDataset.range(num_elements).pipe(
      _TorchRandomMapMapDataset, _with_random_ints, seed=seed
  )


@absltest.skipIf(
    torch_random is None, "PyTorch stateless RNG is not available."
)
class TorchRandomMapMapDatasetTest(parameterized.TestCase):

  def test_transform_receives_cpu_uint64_philox_key(self):
    ds = dataset.MapDataset.range(3).pipe(
        _TorchRandomMapMapDataset,
        lambda element, key: (element, key),
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

    ds = dataset.MapDataset.range(4).pipe(
        _TorchRandomMapMapDataset, split_and_draw, seed=42
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
    with self.assertRaisesRegex(ValueError, r"\[0, 2\*\*64\)"):
      _TorchRandomMapMapDataset(
          dataset.MapDataset.range(4), _with_random_ints, seed=seed
      )

  def test_random_access_matches_iteration(self):
    ds = _make_dataset()
    values = list(ds)
    for index in reversed(range(len(ds))):
      self.assertEqual(ds[index], values[index])

  def test_getitems(self):
    ds = _make_dataset()
    self.assertEqual(ds._getitems([5, 1, 5]), [ds[5], ds[1], ds[5]])

  def test_slice(self):
    ds = _make_dataset()
    self.assertEqual(list(ds[2:5]), [ds[2], ds[3], ds[4]])

  def test_repeat_produces_distinct_epochs(self):
    ds = _make_dataset(num_elements=4).repeat(2)
    items = list(ds)
    epoch_0 = [draws for _, draws in items[:4]]
    epoch_1 = [draws for _, draws in items[4:]]
    self.assertNotEqual(epoch_0, epoch_1)

  def test_seed_from_dataset_seed(self):
    ds = (
        dataset.MapDataset.range(4)
        .seed(7)
        .pipe(_TorchRandomMapMapDataset, _with_random_ints)
    )
    ds_again = (
        dataset.MapDataset.range(4)
        .seed(7)
        .pipe(_TorchRandomMapMapDataset, _with_random_ints)
    )
    self.assertEqual(list(ds), list(ds_again))

  def test_explicit_seed_overrides_dataset_seed(self):
    ds_default = (
        dataset.MapDataset.range(4)
        .seed(7)
        .pipe(_TorchRandomMapMapDataset, _with_random_ints)
    )
    ds_override = (
        dataset.MapDataset.range(4)
        .seed(7)
        .pipe(_TorchRandomMapMapDataset, _with_random_ints, seed=8)
    )
    self.assertNotEqual(list(ds_default), list(ds_override))

  def test_missing_seed_raises(self):
    with self.assertRaisesRegex(ValueError, "requires a seed"):
      _TorchRandomMapMapDataset(dataset.MapDataset.range(4), _with_random_ints)

  def test_callable_class_transform(self):

    class AddUniformNoise:

      def __call__(self, element: int, key: Any) -> float:
        return element + torch_random.uniform(key, ()).item()

    ds = dataset.MapDataset.range(4).pipe(
        _TorchRandomMapMapDataset, AddUniformNoise(), seed=1
    )
    for index, value in enumerate(ds):
      self.assertBetween(value, index, index + 1)
    self.assertIn("AddUniformNoise", str(ds))

  def test_random_map_transform_raises(self):

    class LegacyTransform(transforms.RandomMap):

      def random_map(self, element: int, rng: Any) -> int:
        del rng
        return element

    with self.assertRaisesRegex(TypeError, "expects a callable"):
      dataset.MapDataset.range(4).pipe(
          _TorchRandomMapMapDataset, LegacyTransform(), seed=1
      )

  def test_docstring_example(self):
    def add_noise(x: Any, key: Any) -> Any:
      return x + torch_random.normal(key, x.shape)

    ds = (
        grain.MapDataset.source([torch.zeros(2)] * 4)
        .pipe(grain.experimental.TorchRandomMapMapDataset, add_noise, seed=42)
        .batch(2, batch_fn=torch.utils.data.default_collate)  # pyrefly: ignore[bad-argument-type]
    )
    batch = ds[1]
    self.assertIsNotNone(batch)
    self.assertEqual(batch.shape, (2, 2))
    self.assertTrue(torch.equal(ds[1], ds[1]))
    self.assertFalse(torch.equal(ds[0], ds[1]))

  def test_none_elements_are_not_mapped(self):
    ds = (
        dataset.MapDataset.range(6)
        .filter(lambda x: x % 2 == 0)
        .pipe(
            _TorchRandomMapMapDataset,
            lambda element, key: element + 10,
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

  def test_pickle_round_trip_before_and_after_access(self):
    ds = _make_dataset()
    # Only round-trips an object created by this test.
    restored_before = cloudpickle.loads(cloudpickle.dumps(ds))  # pylint: disable=g-unsafe-pickle-load
    _ = ds[0]
    self.assertIn("_key_for_index", ds.__dict__)
    restored_after = cloudpickle.loads(cloudpickle.dumps(ds))  # pylint: disable=g-unsafe-pickle-load
    self.assertNotIn("_key_for_index", restored_after.__dict__)
    self.assertEqual(list(restored_before), list(ds))
    self.assertEqual(list(restored_after), list(ds))

  def test_iterator_get_and_set_state_preserves_sequence(self):
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

  def test_str(self):
    self.assertStartsWith(
        str(_make_dataset()), "TorchRandomMapMapDataset(transform="
    )

  def test_exported_in_grain_experimental(self):
    self.assertIs(
        grain.experimental.TorchRandomMapMapDataset, _TorchRandomMapMapDataset
    )
    self.assertFalse(hasattr(grain.experimental, "TorchRngPool"))

  def test_torch_is_imported_lazily_on_element_access(self):
    with mock.patch.dict(
        "sys.modules",
        {"torch": None, "torch.func": None, "torch.func._random": None},
    ):
      ds = dataset.MapDataset.range(4).pipe(
          _TorchRandomMapMapDataset, lambda x, key: x, seed=42
      )
      with self.assertRaises(ImportError):
        _ = ds[0]


if __name__ == "__main__":
  absltest.main()
