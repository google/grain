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
"""Testes for tree.py with JAX dependency present."""

from absl.testing import absltest
from grain._src.core import tree
from grain._src.core import tree_test
import jax
import numpy as np


class MyTree:

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __eq__(self, other):
    return self.a == other.a and self.b == other.b


class MyClass:

  def __init__(self, c):
    self.c = c


class TreeJaxTest(tree_test.TreeTest):

  def test_map_custom_tree(self):
    jax.tree_util.register_pytree_node(
        MyTree, lambda t: ((t.a, t.b), None), lambda _, args: MyTree(*args)
    )
    self.assertEqual(
        tree.map_structure(lambda x: x + 1, MyTree(1, 2)), MyTree(2, 3)
    )

  def test_spec_like_with_class(self):
    self.assertEqual(
        tree.spec_like({"B": 1232.4, "C": MyClass(1)}),
        {
            "B": "<class 'float'>[]",
            "C": "<class '__main__.MyClass'>[]",
        },
    )

  def test_spec_like_with_list(self):
    self.assertEqual(
        tree.spec_like({
            "B": 1232.4,
            "C": [
                tree_test.TestClass(a=1, b="v2"),
                tree_test.TestClass(a=2, b="v2"),
            ],
        }),
        {
            "B": "<class 'float'>[]",
            "C": "list<grain._src.core.tree_test.TestClass>[2]",
        },
    )

  def test_spec_like_with_unknown_shape(self):
    self.assertEqual(
        tree.spec_like({
            "B": [np.zeros([2]), np.zeros([1])],
            "C": [],
        }),
        {"B": "list<numpy.ndarray>[unknown shape]", "C": "list<>[0]"},
    )

  def test_spec_like_with_dataclass(self):
    self.assertEqual(
        tree.spec_like(tree_test.TestClass(a=1, b="v2")),
        "<class 'grain._src.core.tree_test.TestClass'>\n"
        "{'a': \"<class 'int'>[]\", 'b': \"<class 'str'>[]\"}[]",
    )


if __name__ == "__main__":
  absltest.main()
