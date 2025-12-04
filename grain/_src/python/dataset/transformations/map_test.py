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
"""Tests for map transformation."""
import dataclasses
import operator
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
from grain._src.core import transforms
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import map as map_ds
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np


class RngPoolTest(absltest.TestCase):

  def test_reset_rng_state(self):
    rng = np.random.Generator(np.random.Philox())
    old_state = rng.bit_generator.state
    rng.integers(10)
    with self.assertRaises(AssertionError):
      np.testing.assert_equal(rng.bit_generator.state, old_state)

    map_ds._reset_rng_state(rng, 0, 0)
    np.testing.assert_equal(rng.bit_generator.state, old_state)


@dataclasses.dataclass(frozen=True)
class MapWithNoTransform(transforms.MapTransform):

  def map(self, element: int):
    return element


@dataclasses.dataclass(frozen=True)
class MapWithTransform(transforms.MapTransform):

  def map(self, element: int):
    return element + 1


class MapWithElementSpecInference(transforms.MapTransform):

  def map(self, element: int):
    return {
        "parsed": np.full((10, 10), element),
        "nested": {"record": str(element)},
    }

  def output_spec(self, input_spec: Any) -> Any:
    assert input_spec == base.ShapeDtypeStruct(
        shape=(), dtype=np.int64
    ), input_spec
    return {
        "parsed": base.ShapeDtypeStruct(shape=(10, 10), dtype=np.int64),
        "nested": {"record": base.ShapeDtypeStruct(shape=(), dtype=str)},
    }


@dataclasses.dataclass(frozen=True)
class RandomMapWithTransform(transforms.RandomMapTransform):

  def random_map(self, element: int, rng: np.random.Generator):
    delta = 0.1
    return element + rng.uniform(-delta, delta)


@dataclasses.dataclass(frozen=True)
class RandomMapWithDeterminismTransform(transforms.RandomMapTransform):

  def random_map(self, element: int, rng: np.random.Generator):
    return element + rng.integers(0, 10)


class RandomMapWithElementSpecInference(transforms.RandomMapTransform):

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


@dataclasses.dataclass(frozen=True)
class AddIndexTransform(transforms.MapWithIndex):

  def map_with_index(self, index: int, element: int):
    return (index, element)


class AddIndexWithElementSpecInference(transforms.MapWithIndex):

  def map_with_index(self, index: int, element: int):
    return (index, element)

  def output_spec(self, input_spec: Any) -> Any:
    assert input_spec == base.ShapeDtypeStruct(
        shape=(), dtype=np.int64
    ), input_spec
    return (
        base.ShapeDtypeStruct(shape=(), dtype=np.int64),
        base.ShapeDtypeStruct(shape=(), dtype=np.int64),
    )


class MapMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)

  def test_map_size(self):
    map_ds_no_transform = map_ds.MapMapDataset(
        self.range_ds, MapWithNoTransform()
    )
    map_ds_with_transform = map_ds.MapMapDataset(
        self.range_ds, MapWithTransform()
    )
    self.assertLen(map_ds_no_transform, len(self.range_ds))
    self.assertLen(map_ds_with_transform, len(self.range_ds))

  @parameterized.parameters(
      MapWithNoTransform,
      MapWithTransform,
  )
  def test_map_picklable(self, map_cls):
    ds = map_ds.MapMapDataset(self.range_ds, map_cls())
    ds = cloudpickle.loads(cloudpickle.dumps(ds))
    self.assertLen(ds, len(self.range_ds))

  def test_map_data_no_transform(self):
    map_ds_no_transform = map_ds.MapMapDataset(
        self.range_ds, MapWithNoTransform()
    )
    expected_data = [i for i in range(10)]
    actual_data = [
        map_ds_no_transform[i] for i in range(len(map_ds_no_transform))
    ]
    self.assertEqual(expected_data, actual_data)

  def test_map_data_with_transform(self):
    map_ds_with_transform = map_ds.MapMapDataset(
        self.range_ds, MapWithTransform()
    )
    expected_data = [i + 1 for i in range(10)]
    actual_data = [
        map_ds_with_transform[i] for i in range(len(map_ds_with_transform))
    ]
    self.assertEqual(expected_data, actual_data)

  @parameterized.parameters(
      dict(
          map_cls=MapWithNoTransform,
          indices=list(range(10)),
          expected_data=list(range(10)),
      ),
      dict(
          map_cls=MapWithNoTransform,
          indices=[0, 4, 8],
          expected_data=[0, 4, 8],
      ),
      dict(
          map_cls=MapWithTransform,
          indices=list(range(10)),
          expected_data=[i + 1 for i in range(10)],
      ),
      dict(
          map_cls=MapWithTransform,
          indices=[0, 4, 8],
          expected_data=[1, 5, 9],
      ),
  )
  def test_map_data_with_get_items(self, map_cls, indices, expected_data):
    ds = map_ds.MapMapDataset(self.range_ds, map_cls())
    actual_data = ds._getitems(indices)
    self.assertEqual(expected_data, actual_data)

  def test_map_checkpointing(self):
    ds = self.range_ds.map(MapWithTransform())
    assert_equal_output_after_checkpoint(ds)

  def test_map_element_spec_inference(self):
    ds = map_ds.MapMapDataset(self.range_ds, MapWithElementSpecInference())
    self.assertEqual(
        ds._element_spec,
        {
            "parsed": base.ShapeDtypeStruct(shape=(10, 10), dtype=np.int64),
            "nested": {"record": base.ShapeDtypeStruct(shape=(), dtype=str)},
        },
    )

  def test_map_element_spec_inference_raises_error(self):
    ds = map_ds.MapMapDataset(self.range_ds, MapWithNoTransform())
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec

  def test_map_element_spec_propagets_output_spec_error(self):
    ds = map_ds.MapMapDataset(
        self.range_ds.batch(2), MapWithElementSpecInference()
    )
    with self.assertRaises(AssertionError):
      _ = ds._element_spec


class RandomMapMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)

  def test_map_size(self):
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    self.assertLen(ds, len(self.range_ds))

  def test_map_picklable(self):
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    ds = cloudpickle.loads(cloudpickle.dumps(ds))
    self.assertLen(ds, len(self.range_ds))

  def test_random_map_data_with_transform(self):
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [ds[i] for i in range(len(ds))]
    np.testing.assert_almost_equal(expected_data, actual_data, decimal=1)

  def test_random_map_with_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed)
    ds1 = map_ds.RandomMapMapDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_ds.seed(seed)
    ds2 = map_ds.RandomMapMapDataset(ds2, RandomMapWithTransform())
    np.testing.assert_almost_equal(list(ds1), list(ds2), decimal=1)

  def test_random_map_overrides_default_seed(self):
    seed = 42
    ds1 = self.range_ds.seed(seed)
    ds1 = map_ds.RandomMapMapDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_ds.seed(seed)
    ds2 = map_ds.RandomMapMapDataset(
        ds2, RandomMapWithTransform(), seed=seed + 1
    )
    np.testing.assert_array_compare(operator.__ne__, list(ds1), list(ds2))

  @parameterized.parameters(
      RandomMapWithTransform(),
      lambda x, rng: x,
  )
  def test_random_map_raises_with_no_seed(self, transform):
    with self.assertRaises(ValueError):
      map_ds.RandomMapMapDataset(self.range_ds, transform)

  @parameterized.parameters(
      dict(indices=list(range(10))),
      dict(indices=[0, 4, 8]),
      dict(indices=[3, 1, 7, 2]),
  )
  def test_random_map_data_with_get_items(self, indices):
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithDeterminismTransform(), seed=42
    )
    expected_data = [ds[i] for i in indices]
    actual_data = ds._getitems(indices)
    np.testing.assert_equal(expected_data, actual_data)

  @parameterized.parameters(0, 1, 42)
  def test_random_map_checkpointing(self, random_map_seed):
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=random_map_seed
    )
    assert_equal_output_after_checkpoint(ds)

  def test_random_map_element_spec_inference(self):
    ds = map_ds.RandomMapMapDataset(
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
    ds = map_ds.RandomMapMapDataset(
        self.range_ds, RandomMapWithTransform(), seed=0
    )
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec


class MapIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(10).to_iter_dataset()

  def test_map_data_no_transform(self):
    map_no_transform_iter_ds = iter(
        map_ds.MapIterDataset(self.range_iter_ds, MapWithNoTransform())
    )
    expected_data = [_ for _ in range(10)]
    actual_data = [next(map_no_transform_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_map_data_with_transform(self):
    map_with_transform_iter_ds = iter(
        map_ds.MapIterDataset(self.range_iter_ds, MapWithTransform())
    )
    expected_data = [i + 1 for i in range(10)]
    actual_data = [next(map_with_transform_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_map_past_one_epoch_raises_exception(self):
    map_no_transform_iter_ds = iter(
        map_ds.MapIterDataset(self.range_iter_ds, MapWithNoTransform())
    )
    with self.assertRaises(StopIteration):
      next(map_no_transform_iter_ds)
      _ = [next(map_no_transform_iter_ds) for _ in range(20)]

  def test_map_checkpointing(self):
    ds = self.range_iter_ds.map(MapWithTransform())
    assert_equal_output_after_checkpoint(ds)

  def test_map_element_spec_inference(self):
    ds = map_ds.MapIterDataset(
        self.range_iter_ds, MapWithElementSpecInference()
    )
    self.assertEqual(
        ds._element_spec,
        {
            "parsed": base.ShapeDtypeStruct(shape=(10, 10), dtype=np.int64),
            "nested": {"record": base.ShapeDtypeStruct(shape=(), dtype=str)},
        },
    )

  def test_map_element_spec_inference_raises_error(self):
    ds = map_ds.MapIterDataset(self.range_iter_ds, MapWithNoTransform())
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec


class RandomMapIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(0, 10).to_iter_dataset()

  def test_random_map_data_with_transform(self):
    map_with_random_transform_iter_ds = iter(
        map_ds.RandomMapIterDataset(
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
          map_ds.RandomMapIterDataset(
              self.range_iter_ds, RandomMapWithDeterminismTransform(), seed=42
          )
      )
      actual_data = [next(map_with_random_transform_iter_ds) for _ in range(10)]
      np.testing.assert_equal(expected_data, actual_data)

  def test_random_map_with_default_seed(self):
    seed = 42
    ds1 = self.range_iter_ds.seed(seed)
    ds1 = map_ds.RandomMapIterDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_iter_ds.seed(seed)
    ds2 = map_ds.RandomMapIterDataset(ds2, RandomMapWithTransform())
    np.testing.assert_almost_equal(list(ds1), list(ds2), decimal=1)

  def test_random_map_overrides_default_seed(self):
    seed = 42
    ds1 = self.range_iter_ds.seed(seed)
    ds1 = map_ds.RandomMapIterDataset(ds1, RandomMapWithTransform())
    ds2 = self.range_iter_ds.seed(seed)
    ds2 = map_ds.RandomMapIterDataset(
        ds2, RandomMapWithTransform(), seed=seed + 1
    )
    np.testing.assert_array_compare(operator.__ne__, list(ds1), list(ds2))

  @parameterized.parameters(
      RandomMapWithTransform(),
      lambda x, rng: x,
  )
  def test_random_map_raises_with_no_seed(self, transform):
    with self.assertRaises(ValueError):
      map_ds.RandomMapIterDataset(self.range_iter_ds, transform)

  @parameterized.parameters(0, 1, 42)
  def test_random_map_checkpointing(self, random_map_seed):
    ds = map_ds.RandomMapIterDataset(
        self.range_iter_ds, RandomMapWithTransform(), seed=random_map_seed
    )
    assert_equal_output_after_checkpoint(ds)

  def test_random_map_element_spec_inference(self):
    ds = map_ds.RandomMapIterDataset(
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
    ds = map_ds.RandomMapIterDataset(
        self.range_iter_ds, RandomMapWithTransform(), seed=0
    )
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec


class MapWithIndexMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(3, 6)

  def test_length(self):
    ds = map_ds.MapWithIndexMapDataset(self.range_ds, AddIndexTransform())
    self.assertLen(ds, len(self.range_ds))

  def test_getitem(self):
    ds = map_ds.MapWithIndexMapDataset(self.range_ds, AddIndexTransform())
    self.assertEqual(ds[0], (0, 3))
    self.assertEqual(ds[1], (1, 4))
    self.assertEqual(ds[2], (2, 5))
    self.assertEqual(ds[3], (3, 3))
    self.assertEqual(ds[4], (4, 4))
    self.assertEqual(ds[5], (5, 5))

  @parameterized.parameters(
      dict(indices=list(range(3, 6))),
      dict(indices=[3, 4, 5]),
      dict(indices=[4, 3, 5]),
  )
  def test_map_with_index_data_with_get_items(self, indices):
    ds = map_ds.MapWithIndexMapDataset(self.range_ds, AddIndexTransform())
    expected_data = [ds[i] for i in indices]
    actual_data = ds._getitems(indices)
    self.assertEqual(expected_data, actual_data)

  def test_map_with_index_element_spec_inference(self):
    ds = map_ds.MapWithIndexMapDataset(
        self.range_ds, AddIndexWithElementSpecInference()
    )
    self.assertEqual(
        ds._element_spec,
        (
            base.ShapeDtypeStruct(shape=(), dtype=np.int64),
            base.ShapeDtypeStruct(shape=(), dtype=np.int64),
        ),
    )

  def test_map_with_index_element_spec_inference_raises_error(self):
    ds = map_ds.MapWithIndexMapDataset(self.range_ds, AddIndexTransform())
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec


class MapWithIndexIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(10).to_iter_dataset()

  def test_map_with_index_transform(self):
    map_with_index_transform_iter_ds = iter(
        map_ds.MapWithIndexIterDataset(self.range_iter_ds, AddIndexTransform())
    )
    expected_data = [(i, i) for i in range(10)]
    actual_data = [next(map_with_index_transform_iter_ds) for _ in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_map_with_index_past_one_epoch_raises_exception(self):
    map_with_index_transform_iter_ds = iter(
        map_ds.MapWithIndexIterDataset(self.range_iter_ds, AddIndexTransform())
    )
    with self.assertRaises(StopIteration):
      next(map_with_index_transform_iter_ds)
      _ = [next(map_with_index_transform_iter_ds) for _ in range(20)]

  def test_map_with_index_checkpointing(self):
    ds = self.range_iter_ds.map_with_index(AddIndexTransform())
    assert_equal_output_after_checkpoint(ds)

  def test_map_with_index_element_spec_inference(self):
    ds = map_ds.MapWithIndexIterDataset(
        self.range_iter_ds, AddIndexWithElementSpecInference()
    )
    self.assertEqual(
        ds._element_spec,
        (
            base.ShapeDtypeStruct(shape=(), dtype=np.int64),
            base.ShapeDtypeStruct(shape=(), dtype=np.int64),
        ),
    )

  def test_map_with_index_element_spec_inference_raises_error(self):
    ds = map_ds.MapWithIndexIterDataset(self.range_iter_ds, AddIndexTransform())
    with self.assertRaisesRegex(ValueError, "does not implement `output_spec`"):
      _ = ds._element_spec


if __name__ == "__main__":
  absltest.main()
