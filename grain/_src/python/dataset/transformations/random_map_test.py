# Copyright 2023 Google LLC
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
"""Tests for random map transformation."""

import dataclasses
import operator
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
from grain._src.core import transforms
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import random_map as random_map_ds
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np


class RngPoolTest(absltest.TestCase):

  def test_reset_rng_state(self):
    rng = np.random.Generator(np.random.Philox())
    old_state = rng.bit_generator.state
    rng.integers(10)
    with self.assertRaises(AssertionError):
      np.testing.assert_equal(rng.bit_generator.state, old_state)

    random_map_ds._reset_rng_state(rng, 0, 0)
    np.testing.assert_equal(rng.bit_generator.state, old_state)

  def test_example_docstring(self):
    class ContrastiveMapDataset(dataset.MapDataset):

      def __init__(self, parent: dataset.MapDataset, seed: int):
        super().__init__(parent)
        self._rng_pool = random_map_ds.RngPool(seed)

      def __len__(self) -> int:
        return len(self._parent)

      def __getitem__(self, index):
        if isinstance(index, slice):
          return self.slice(index)
        rng = self._rng_pool.acquire_rng(index)
        other_index = int(rng.integers(0, len(self._parent) - 1))
        if other_index >= index:
          other_index += 1
        self._rng_pool.release_rng(rng)
        return (self._parent[index], self._parent[other_index])

    parent_ds = dataset.MapDataset.source([10, 20, 30, 40, 50, 60, 70])
    paired_ds = ContrastiveMapDataset(parent_ds, seed=42)

    self.assertEqual(
        list(paired_ds),
        [(10, 70), (20, 30), (30, 40), (40, 60), (50, 10), (60, 20), (70, 30)],
    )
    self.assertEqual(paired_ds[2], (30, 40))


@dataclasses.dataclass(frozen=True)
class RandomMapWithTransform(transforms.RandomMap):

  def random_map(self, element: int, rng: np.random.Generator):
    delta = 0.1
    return element + rng.uniform(-delta, delta)


@dataclasses.dataclass(frozen=True)
class RandomMapWithDeterminismTransform(transforms.RandomMap):

  def random_map(self, element: int, rng: np.random.Generator):
    return element + rng.integers(0, 10)


class RandomMapWithElementSpecInference(transforms.RandomMap):

  def random_map(self, element: int, rng: np.random.Generator):
    return {
        "nested": {"record": element + rng.integers(0, 10)},
    }

  def output_spec(self, input_spec: Any) -> Any:
    assert input_spec == base.ShapeDtypeStruct(
        shape=(), dtype=np.int64
    ), input_spec
    return {
        "nested": {"record": base.ShapeDtypeStruct(shape=(), dtype=np.int64)},
    }


class RandomMapMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)

  def test_map_size(self):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    self.assertLen(ds, len(self.range_ds))

  def test_map_picklable(self):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    ds = cloudpickle.loads(cloudpickle.dumps(ds))  # pylint: disable=g-unsafe-pickle-load
    self.assertLen(ds, len(self.range_ds))

  def test_random_map_data_with_transform(self):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [ds[i] for i in range(len(ds))]
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)

  def test_random_map_with_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed)
    ds1 = random_map_ds.RandomMapMapDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_ds.seed(seed)
    ds2 = random_map_ds.RandomMapMapDataset(ds2, RandomMapWithTransform())
    np.testing.assert_almost_equal(list(ds1), list(ds2), decimal=1)

  def test_random_map_overrides_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed)
    ds1 = random_map_ds.RandomMapMapDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_ds.seed(seed)
    ds2 = random_map_ds.RandomMapMapDataset(
        ds2, RandomMapWithTransform(), seed=seed + 1
    )
    np.testing.assert_array_compare(operator.__ne__, list(ds1), list(ds2))

  @parameterized.parameters(
      RandomMapWithTransform(),
      lambda x, rng: x,
  )
  def test_random_map_raises_with_no_seed(self, transform):
    with self.assertRaises(ValueError):
      random_map_ds.RandomMapMapDataset(self.range_ds, transform)

  @parameterized.parameters(
      dict(indices=list(range(10))),
      dict(indices=[0, 4, 8]),
      dict(indices=[3, 1, 7, 2]),
  )
  def test_random_map_data_with_get_items(self, indices):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithDeterminismTransform(), seed=42
    )
    expected_data = [ds[i] for i in indices]
    actual_data = ds._getitems(indices)
    np.testing.assert_equal(expected_data, actual_data)

  @parameterized.parameters(0, 1, 42)
  def test_random_map_checkpointing(self, random_map_seed):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=random_map_seed
    )
    assert_equal_output_after_checkpoint(ds)

  def test_random_map_element_spec_inference(self):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithElementSpecInference(), seed=0
    )
    self.assertEqual(
        ds._element_spec,
        {
            "nested": {
                "record": base.ShapeDtypeStruct(shape=(), dtype=np.int64)
            },
        },
    )

  def test_random_map_element_spec_inference_raises_error(self):
    ds = random_map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec

  def test_cross_version_determinism(self):
    # This test validates random map determinism across different versions of
    # Grain given a fixed seed. Note that we technically do not guarantee
    # cross-version determinism because numpy does not. Any changes to the numpy
    # RNGs or how we use them could break this. Multiple users nevertheless rely
    # on it because it holds in practice. Only update the values if you know
    # what you're doing.
    ds = dataset.MapDataset.range(10).seed(41)
    ds = random_map_ds.RandomMapMapDataset(
        ds, RandomMapWithDeterminismTransform()
    )
    ds = ds.map(lambda x: x.item())
    self.assertEqual(list(ds), [0, 6, 9, 5, 10, 14, 14, 13, 15, 13])


class RandomMapIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(0, 10).to_iter_dataset()

  def test_random_map_data_with_transform(self):
    map_with_random_transform_iter_ds = iter(
        random_map_ds.RandomMapIterDataset(
            self.range_iter_ds, RandomMapWithTransform(), seed=0
        )
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(map_with_random_transform_iter_ds) for _ in range(10)]
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)

  def test_random_map_data_with_transform_deterministic_with_seed(self):
    expected_data = [9, 3, 6, 11, 4, 7, 9, 9, 8, 9]
    # check if elements are reproducible for multiple runs
    for _ in range(10):
      map_with_random_transform_iter_ds = iter(
          random_map_ds.RandomMapIterDataset(
              self.range_iter_ds, RandomMapWithDeterminismTransform(), seed=42
          )
      )
      actual_data = [next(map_with_random_transform_iter_ds) for _ in range(10)]
      np.testing.assert_equal(expected_data, actual_data)

  def test_random_map_with_default_seed(self):
    seed = 42
    ds1 = self.range_iter_ds.seed(seed)
    ds1 = random_map_ds.RandomMapIterDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_iter_ds.seed(seed)
    ds2 = random_map_ds.RandomMapIterDataset(ds2, RandomMapWithTransform())
    np.testing.assert_almost_equal(list(ds1), list(ds2), decimal=1)

  def test_random_map_overrides_default_seed(self):
    seed = 42
    ds1 = self.range_iter_ds.seed(seed)
    ds1 = random_map_ds.RandomMapIterDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_iter_ds.seed(seed)
    ds2 = random_map_ds.RandomMapIterDataset(
        ds2, RandomMapWithTransform(), seed=seed + 1
    )
    np.testing.assert_array_compare(operator.__ne__, list(ds1), list(ds2))

  @parameterized.parameters(
      RandomMapWithTransform(),
      lambda x, rng: x,
  )
  def test_random_map_raises_with_no_seed(self, transform):
    with self.assertRaises(ValueError):
      random_map_ds.RandomMapIterDataset(self.range_iter_ds, transform)

  @parameterized.parameters(0, 1, 42)
  def test_random_map_checkpointing(self, random_map_seed):
    ds = random_map_ds.RandomMapIterDataset(
        self.range_iter_ds, RandomMapWithTransform(), seed=random_map_seed
    )
    assert_equal_output_after_checkpoint(ds)

  def test_random_map_element_spec_inference(self):
    ds = random_map_ds.RandomMapIterDataset(
        self.range_iter_ds, RandomMapWithElementSpecInference(), seed=0
    )
    self.assertEqual(
        ds._element_spec,
        {
            "nested": {
                "record": base.ShapeDtypeStruct(shape=(), dtype=np.int64)
            },
        },
    )

  def test_random_map_element_spec_inference_raises_error(self):
    ds = random_map_ds.RandomMapIterDataset(
        self.range_iter_ds, RandomMapWithTransform(), seed=0
    )
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec

  def test_get_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = random_map_ds.RandomMapIterDataset(
        ds, RandomMapWithTransform(), seed=0
    )
    ds_iter = mapped_ds.__iter__()
    for i in range(20):
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      _ = next(ds_iter)

  def test_set_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = random_map_ds.RandomMapIterDataset(
        ds, RandomMapWithTransform(), seed=0
    )
    expected = list(mapped_ds)
    ds_iter = mapped_ds.__iter__()
    for i in reversed(range(20)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), expected[i])


class _ToyStatelessRandomMap:

  def make_key_for_index(self, seed: int):
    return lambda index: (seed * 1000) + index

  def stateless_random_map(self, element: int, key: int) -> int:
    return element + key


class _ToyStatelessRandomMapWithSpec(_ToyStatelessRandomMap):

  def output_spec(self, input_spec: Any) -> Any:
    return {"val": input_spec}

  def stateless_random_map(self, element: int, key: int) -> dict[str, int]:  # pyrefly: ignore[bad-override]
    return {"val": element + key}


class StatelessRandomMapDatasetTest(parameterized.TestCase):

  def test_map_dataset_stateless_random_map_and_apply(self):
    ds = dataset.MapDataset.range(4).stateless_random_map(
        _ToyStatelessRandomMap(), seed=42
    )
    self.assertEqual(list(ds), [42000, 42002, 42004, 42006])

    applied = (
        dataset.MapDataset.range(4).seed(43).apply(_ToyStatelessRandomMap())
    )
    direct = (
        dataset.MapDataset.range(4)
        .seed(43)
        .stateless_random_map(_ToyStatelessRandomMap())
    )
    self.assertEqual(list(applied), list(direct))

  def test_iter_dataset_stateless_random_map_checkpointing_and_seeking(self):
    iter_ds = (
        dataset.MapDataset.range(10)
        .to_iter_dataset()
        .stateless_random_map(_ToyStatelessRandomMap(), seed=42)
    )
    assert_equal_output_after_checkpoint(iter_ds)

    expected = list(iter_ds)
    ds_iter = iter_ds.__iter__()
    for i in reversed(range(10)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      self.assertEqual(next(ds_iter), expected[i])

  def test_stateless_random_map_element_spec(self):
    map_ds = dataset.MapDataset.range(4).stateless_random_map(
        _ToyStatelessRandomMapWithSpec(), seed=42
    )
    iter_ds = (
        dataset.MapDataset.range(4)
        .to_iter_dataset()
        .stateless_random_map(_ToyStatelessRandomMapWithSpec(), seed=42)
    )
    expected_spec = {
        "val": base.ShapeDtypeStruct(shape=(), dtype=np.int64),
    }
    self.assertEqual(map_ds._element_spec, expected_spec)
    self.assertEqual(iter_ds._element_spec, expected_spec)

  def test_plain_callable_defaults_to_jax_stateless_without_eager_import(self):
    ds = random_map_ds.StatelessRandomMapMapDataset(
        dataset.MapDataset.range(4), lambda x, key: x, seed=42
    )
    self.assertIsInstance(
        ds._stateless_transform, transforms.JaxStatelessRandomMap  # pylint: disable=protected-access
    )

  def test_rejects_non_stateless_transform(self):
    with self.assertRaisesRegex(TypeError, "StatelessRandomMap"):
      dataset.MapDataset.range(4).stateless_random_map(
          RandomMapWithTransform(), seed=42  # pyrefly: ignore[bad-argument-type]
      )
    with self.assertRaisesRegex(TypeError, "StatelessRandomMap"):
      dataset.MapDataset.range(4).to_iter_dataset().stateless_random_map(
          RandomMapWithTransform(), seed=42  # pyrefly: ignore[bad-argument-type]
      )


if __name__ == "__main__":
  absltest.main()
