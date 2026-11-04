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


@dataclasses.dataclass(frozen=True)
class MapWithNoTransform(transforms.Map):

  def map(self, element: int):
    return element


@dataclasses.dataclass(frozen=True)
class MapWithTransform(transforms.Map):

  def map(self, element: int):
    return element + 1


class MapWithElementSpecInference(transforms.Map):

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

  def test_get_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = map_ds.MapIterDataset(ds, MapWithNoTransform())
    ds_iter = mapped_ds.__iter__()
    for i in range(20):
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      _ = next(ds_iter)

  def test_set_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = map_ds.MapIterDataset(ds, MapWithNoTransform())
    ds_iter = mapped_ds.__iter__()
    for i in reversed(range(20)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), i)


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

  def test_get_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = map_ds.MapWithIndexIterDataset(ds, AddIndexTransform())
    ds_iter = mapped_ds.__iter__()
    for i in range(20):
      self.assertEqual(dataset.get_next_index(ds_iter), i)
      _ = next(ds_iter)

  def test_set_next_index(self):
    ds = dataset.MapDataset.range(0, 20).to_iter_dataset()
    mapped_ds = map_ds.MapWithIndexIterDataset(ds, AddIndexTransform())
    ds_iter = mapped_ds.__iter__()
    for i in reversed(range(20)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), (i, i))


if __name__ == "__main__":
  absltest.main()
