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


class _TestFilter(transforms.Filter):

  def filter(self, x):
    return x % 2 == 0


class _TestFilterWithStr(transforms.Filter):

  def filter(self, x):
    return x % 2 == 0

  def __str__(self):
    return "CustomStr"


class _TestMapWithRepr(transforms.MapTransform):

  def map(self, x):
    return x % 2 == 0

  def __repr__(self):
    return "CustomRepr"


def add_one(x):
  return x + 1


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
          transform=transforms.MapTransform.from_callable(add_one),
          expected_substring=(
              "MapFromCallable<add_one @ .../_src/core/transforms_test.py:"
          ),
      ),
      dict(
          transform=transforms.MapTransform.from_callable(lambda x: x + 1),
          expected_substring=(
              "MapFromCallable<<lambda> @ .../_src/core/transforms_test.py:"
          ),
      ),
  )
  def test_get_pretty_transform_name(self, transform, expected_substring):
    self.assertIn(
        expected_substring, transforms.get_pretty_transform_name(transform)
    )


class MapFromCallableTest(parameterized.TestCase):

  def test_local_function(self):
    map_transform = transforms.MapTransform.from_callable(add_one)
    self.assertEqual(map_transform.map(1), 2)

  def test_lambda(self):
    map_transform = transforms.MapTransform.from_callable(lambda x: x + 1)
    self.assertEqual(map_transform.map(1), 2)


if __name__ == "__main__":
  absltest.main()
