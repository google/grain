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
"""Utilities for working with pytrees.

See https://jax.readthedocs.io/en/latest/pytrees.html for more details about
pytrees.

This module merely re-directs imports of the actual implementations. To avoid a
direct dependency on JAX, we check if it's already present and resort to the
`tree` package otherwise.

We should be able to remove this module once b/257971667 is resolved.
"""

try:
  from jax import tree_util  # pytype: disable=import-error # pylint: disable=g-import-not-at-top

  map_structure = tree_util.tree_map
  map_structure_with_path = tree_util.tree_map_with_path

  def assert_same_structure(a, b):
    a_structure = tree_util.tree_structure(a)
    b_structure = tree_util.tree_structure(b)
    if a_structure != b_structure:
      raise ValueError(
          f"Structures are not the same: a = {a_structure}, b = {b_structure}"
      )

  def flatten(structure):
    return tree_util.tree_flatten(structure)[0]

  def unflatten_as(structure, flat_sequence):
    return tree_util.tree_unflatten(
        tree_util.tree_structure(structure), flat_sequence
    )

except ImportError:
  import tree  # pylint: disable=g-import-not-at-top

  map_structure = tree.map_structure
  map_structure_with_path = tree.map_structure_with_path
  assert_same_structure = tree.assert_same_structure
  flatten = tree.flatten
  unflatten_as = tree.unflatten_as
