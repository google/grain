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
"""Testes for tree_lib.py.

Since the tree_lib.py only re-directs the actual implementations this test does
not try to cover the actual functionality, but rather the re-direction
correctness.
"""

import dataclasses
from typing import Protocol, runtime_checkable

from absl.testing import absltest
from absl.testing import parameterized
import attrs
from grain._src.core import tree_lib
import numpy as np


@runtime_checkable
class TreeImpl(Protocol):

  def map_structure(self, f, *structures):
    ...

  def map_structure_with_path(self, f, *structures):
    ...

  def assert_same_structure(self, a, b):
    ...

  def flatten(self, structure):
    ...

  def flatten_with_path(self, structure):
    ...

  def unflatten_as(self, structure, flat_sequence):
    ...

  def spec_like(self, structure):
    ...


# Static check that the module implements the necessary functions.
tree_lib: TreeImpl = tree_lib


@dataclasses.dataclass
class _TestClass:
  a: int
  b: str


class MyClass:

  def __init__(self, c):
    self.c = c


@attrs.define
class MyAttrs:
  d: int
  e: str


class TreeTest(parameterized.TestCase):

  def test_implements_tree_protocol(self):
    # Run time check that the module implements the necessary functions.
    # The module impl branching happens at run time, so the static check does
    # not cover both branches.
    self.assertIsInstance(tree_lib, TreeImpl)

  def test_map_structure(self):
    self.assertEqual(
        tree_lib.map_structure(
            lambda x: x + 1, ({"B": 10, "A": 20}, [1, 2], 3)
        ),
        ({"B": 11, "A": 21}, [2, 3], 4),
    )

  def test_map_structure_with_path(self):
    self.assertEqual(
        tree_lib.map_structure_with_path(
            lambda path, x: x if path else None, {"B": "v1", "A": "v2"}
        ),
        {"B": "v1", "A": "v2"},
    )

  def test_assert_same_structure(self):
    tree_lib.assert_same_structure({"B": "v1", "A": "v2"}, {"B": 10, "A": 20})

  def test_flatten(self):
    self.assertEqual(tree_lib.flatten({"A": "v2", "B": "v1"}), ["v2", "v1"])

  def test_flatten_with_path(self):
    result = tree_lib.flatten_with_path({"A": "v2", "B": "v1"})
    # Maybe extract keys from path elements.
    result = tree_lib.map_structure(lambda x: getattr(x, "key", x), result)
    self.assertEqual(result, [(("A",), "v2"), (("B",), "v1")])

  def test_unflatten_as(self):
    self.assertEqual(
        tree_lib.unflatten_as({"A": "v2", "B": "v1"}, [1, 2]), {"A": 1, "B": 2}
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="simple",
          structure={"A": "v2", "B": 1232.4, "C": np.ndarray([1, 2, 3])},
          expected_output={
              "A": "str[]",
              "B": "float[]",
              "C": "float64[1, 2, 3]",
          },
      ),
      dict(
          testcase_name="nested",
          structure={"A": "v2", "B": {"C": np.ndarray([1, 2, 3])}},
          expected_output={
              "A": "str[]",
              "B": {"C": "float64[1, 2, 3]"},
          },
      ),
      dict(
          testcase_name="leaf",
          structure=np.ndarray([1, 2, 3]),
          expected_output="float64[1, 2, 3]",
      ),
  )
  def test_spec_like(self, structure, expected_output):
    self.assertEqual(tree_lib.spec_like(structure), expected_output)

  def test_spec_like_with_class(self):
    self.assertIn(
        tree_lib.spec_like({"B": 1232.4, "C": MyClass(1)}),
        [
            {
                "B": "float[]",
                "C": "MyClass[]",
            },
        ],
    )

  def test_spec_like_with_list(self):
    self.assertEqual(
        tree_lib.spec_like({
            "B": 1232.4,
            "C": [
                _TestClass(a=1, b="v2"),
                _TestClass(a=2, b="v2"),
            ],
        }),
        {
            "B": "float[]",
            "C": "list<_TestClass>[2]",
        },
    )

  def test_spec_like_with_dataclass(self):
    self.assertEqual(
        tree_lib.spec_like(_TestClass(a=1, b="v2")),
        "_TestClass\n{'a': 'int[]', 'b': 'str[]'}[]",
    )

  def test_spec_like_with_unknown_shape(self):
    self.assertEqual(
        tree_lib.spec_like({
            "B": [np.zeros([2]), np.zeros([1])],
            "C": [],
        }),
        {"B": "list<ndarray>[unknown shape]", "C": "list<>[0]"},
    )

  def test_spec_like_with_attrs(self):
    self.assertIn(
        tree_lib.spec_like(MyAttrs(d=1, e="v2")),
        [
            "MyAttrs\n{'d': 'int[]', 'e': 'str[]'}[]",
            "MyAttrs\n{'d': 'int[]', 'e': 'str[]'}[]",
        ],
    )


if __name__ == "__main__":
  absltest.main()
