# Copyright 2024 Google LLC
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
"""Tests for base.py."""

import platform
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python import data_sources
from grain._src.python.dataset import base
import jax
import numpy as np


@absltest.skipIf(
    platform.system() == "Windows", "ArrayRecord isn't supported on Windows."
)
class RandomAccessDataSourceTest(parameterized.TestCase):

  @parameterized.parameters(
      data_sources.ArrayRecordDataSource,
      data_sources.RangeDataSource,
      data_sources.SharedMemoryDataSource,
  )
  def test_protocol(self, source_cls):
    self.assertIsInstance(source_cls, base.RandomAccessDataSource)

  def test_example_docstring(self):
    class MyInMemorySource:

      def __init__(self, data: list[str]):
        self._data = data

      def __len__(self) -> int:
        return len(self._data)

      def __getitem__(self, index: int):
        return self._data[index]

      def __repr__(self) -> str:
        return f"MyInMemorySource(size={len(self)})"

    source = MyInMemorySource(["a", "b", "c"])
    self.assertIsInstance(source, base.RandomAccessDataSource)


class DatasetSelectionMapTest(parameterized.TestCase):

  def test_example_docstring(self):
    class ConcatMap(base.DatasetSelectionMap):

      def __len__(self) -> int:
        return 5

      def __getitem__(self, index: int) -> tuple[int, int]:
        if index < 2:
          return (0, index)
        else:
          return (1, index - 2)

    cmap = ConcatMap()
    self.assertLen(cmap, 5)
    self.assertEqual(cmap[3], (1, 1))


class DatasetOptionsTest(parameterized.TestCase):

  def test_example_docstring_for_merge(self):
    opt1 = base.DatasetOptions(min_shm_size=1024)
    opt2 = base.DatasetOptions(
        min_shm_size=512, filter_warn_threshold_ratio=None
    )
    merged = opt1.merge(opt2)
    self.assertEqual(merged.min_shm_size, 1024)
    self.assertIsNone(merged.filter_warn_threshold_ratio)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_conflicts",
          a=base.DatasetOptions(filter_warn_threshold_ratio=0.1),
          b=base.DatasetOptions(filter_raise_threshold_ratio=0.2),
          expected=base.DatasetOptions(
              filter_warn_threshold_ratio=0.1,
              filter_raise_threshold_ratio=0.2,
          ),
      ),
      dict(
          testcase_name="all_fields_default",
          a=base.DatasetOptions(),
          b=base.DatasetOptions(
              filter_warn_threshold_ratio=0.4,
              filter_raise_threshold_ratio=0.3,
          ),
          expected=base.DatasetOptions(
              filter_warn_threshold_ratio=0.4,
              filter_raise_threshold_ratio=0.3,
          ),
      ),
      dict(
          testcase_name="field_conflict",
          a=base.DatasetOptions(filter_raise_threshold_ratio=0.1),
          b=base.DatasetOptions(filter_raise_threshold_ratio=0.2),
          expected=base.DatasetOptions(
              filter_raise_threshold_ratio=0.1,
          ),
      ),
  )
  def test_merge(self, a, b, expected):
    self.assertEqual(a.merge(b), expected)


class IteratorContextTest(parameterized.TestCase):

  def test_merge(self):
    a = base.IteratorContext(
        dataset_options=base.DatasetOptions(filter_warn_threshold_ratio=0.1)
    )
    b = base.IteratorContext(
        dataset_options=base.DatasetOptions(
            filter_warn_threshold_ratio=0.2, filter_raise_threshold_ratio=0.2
        )
    )
    a.merge(b)
    self.assertEqual(
        a,
        base.IteratorContext(
            dataset_options=base.DatasetOptions(
                filter_warn_threshold_ratio=0.1,
                filter_raise_threshold_ratio=0.2,
            )
        ),
    )

  def test_merge_with_different_mp_context(self):
    a = base.IteratorContext(
        mp_context=base.MultiprocessingContext(process_index=0, process_count=1)
    )
    b = base.IteratorContext(
        mp_context=base.MultiprocessingContext(process_index=1, process_count=2)
    )
    with self.assertRaisesRegex(
        ValueError, "Cannot merge contexts from different worker processes"
    ):
      a.merge(b)


class ShapeDtypeStructTest(parameterized.TestCase):

  def test_shape_dtype_struct_implements_protocol(self):
    self.assertIsInstance(
        base.ShapeDtypeStruct(shape=(1, 2), dtype=np.float32),
        base.ShapeDtypeStructProtocol,
    )

  def test_jax_shape_dtype_struct_implements_protocol(self):
    self.assertIsInstance(
        jax.ShapeDtypeStruct(shape=(1, 2), dtype=np.float32),
        base.ShapeDtypeStructProtocol,
    )


if __name__ == "__main__":
  absltest.main()
