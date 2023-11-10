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


class MyTree:

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __eq__(self, other):
    return self.a == other.a and self.b == other.b


class TreeJaxTest(tree_test.TreeTest):

  def test_map_custom_tree(self):
    jax.tree_util.register_pytree_node(
        MyTree, lambda t: ((t.a, t.b), None), lambda _, args: MyTree(*args)
    )
    self.assertEqual(
        tree.map_structure(lambda x: x + 1, MyTree(1, 2)), MyTree(2, 3)
    )


if __name__ == "__main__":
  absltest.main()
