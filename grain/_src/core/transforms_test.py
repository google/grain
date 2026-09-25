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

import functools
import platform
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
from grain._src.python.dataset import dataset


class _TestFilter(transforms.Filter):

  def filter(self, x):
    return x % 2 == 0


class _TestFilterWithStr(transforms.Filter):

  def filter(self, x):
    return x % 2 == 0

  def __str__(self):
    return "CustomStr"


class _TestMapWithRepr(transforms.Map):

  def map(self, x):
    return x % 2 == 0

  def __repr__(self):
    return "CustomRepr"


@absltest.skipIf(
    platform.system() == "Windows", "Skipped due to windows paths."
)
class GetPrettyTransformNameTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          transform=lambda x: x,
          expected_substring="<lambda> @ .../_src/core/transforms_test.py:",
      ),
      dict(
          transform=transforms.get_pretty_transform_name,
          expected_substring=(
              "get_pretty_transform_name @ .../_src/core/transforms.py:"
          ),
      ),
      dict(transform=list, expected_substring="list"),
      dict(
          transform=functools.partial(lambda x, y: x + y, 1),
          expected_substring="functools.partial",
      ),
      dict(transform=_TestFilter(), expected_substring="_TestFilter"),
      dict(
          transform=_TestFilterWithStr(),
          expected_substring="CustomStr",
      ),
      dict(
          transform=_TestMapWithRepr(),
          expected_substring="CustomRepr",
      ),
  )
  def test_get_pretty_transform_name(self, transform, expected_substring):
    self.assertIn(
        expected_substring, transforms.get_pretty_transform_name(transform)
    )


class MapwithIndexTest(parameterized.TestCase):

  def test_docstring_example(self):
    class AddIndex(transforms.MapWithIndex):

      def map_with_index(self, index: int, element: int) -> int:
        return index + element * 10

    parent_ds = dataset.MapDataset.range(3)
    self.assertEqual(list(parent_ds), [0, 1, 2])
    transformed_ds = parent_ds.map_with_index(AddIndex())
    self.assertEqual(list(transformed_ds), [0, 11, 22])


class FilterTest(parameterized.TestCase):

  def test_docstring_example(self):
    class KeepEven(transforms.Filter):

      def filter(self, element: int) -> bool:
        return element % 2 == 0

    parent_ds = dataset.MapDataset.range(10)
    self.assertEqual(list(parent_ds), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    transformed_ds = parent_ds.filter(KeepEven())
    self.assertEqual(list(transformed_ds), [0, 2, 4, 6, 8])


class MapandRandomMapTest(parameterized.TestCase):

  def test_map_docstring_example(self):

    class AddOne(transforms.Map):

      def map(self, element: int) -> int:
        return element + 1

    parent_ds = dataset.MapDataset.range(5)
    self.assertEqual(list(parent_ds), [0, 1, 2, 3, 4])
    transformed_ds = parent_ds.map(AddOne())
    self.assertEqual(list(transformed_ds), [1, 2, 3, 4, 5])

  def test_random_map_docstring_example(self):
    class AddRandomOffset(transforms.RandomMap):

      def random_map(self, element: int, rng) -> int:
        return element + int(rng.integers(0, 10))

    parent_ds = dataset.MapDataset.range(5).seed(42)
    self.assertEqual(list(parent_ds), [0, 1, 2, 3, 4])
    transformed_ds = parent_ds.random_map(AddRandomOffset())
    self.assertEqual(list(transformed_ds), [2, 5, 5, 4, 9])


if __name__ == "__main__":
  absltest.main()
