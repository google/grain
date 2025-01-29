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
import dataclasses
import pprint

import numpy as np

try:
  # `attrs` is not a native package, avoid explicit dependency and only check if
  # it's present.
  import attrs  # pylint: disable=g-import-not-at-top

  def _is_attrs(obj):
    return attrs.has(obj)

  def _attrs_fields(obj):
    result = {}
    for field in attrs.fields(obj.__class__):
      result[field.name] = getattr(obj, field.name)
    return result

except ImportError:
  attrs = None

  def _is_attrs(unused_obj):
    return False

  def _attrs_fields(unused_obj):
    return {}


try:
  from jax import tree_util  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

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

  def flatten_with_path(structure):
    return tree_util.tree_flatten_with_path(structure)[0]

  def unflatten_as(structure, flat_sequence):
    return tree_util.tree_unflatten(
        tree_util.tree_structure(structure), flat_sequence
    )

  def spec_like(structure):
    """Infers specification of a tree structure.

    Args:
      structure: The structure to get the spec of.

    Returns:
      Same structure but with the leaves replaced with their stringified spec.
      Homogeneous lists and tuples are represented as `container<inner_type>`.
    """
    def _is_leaf(element):
      return isinstance(element, (list, tuple)) and all(
          isinstance(item, type(element[0])) for item in element
      )

    def _type(obj):
      if isinstance(obj, np.ndarray):
        return np.asarray(obj).dtype
      elif isinstance(obj, (list, tuple)) and all(
          isinstance(item, type(obj[0])) for item in obj
      ):
        container_type = type(obj).__name__
        if obj:
          inner_type = f"{type(obj[0]).__module__}.{type(obj[0]).__name__}"
        else:
          inner_type = ""
        return f"{container_type}<{inner_type}>"
      elif dataclasses.is_dataclass(obj):
        # We avoid registering the dataclass with jax's tree_util because
        # that would affect the behavior outside of this module. We also avoid
        # using dataclasses.asdict() since it makes data copy.
        fields = {}
        for field in dataclasses.fields(obj):
          fields[field.name] = getattr(obj, field.name)
        return f"{type(obj)}\n{pprint.pformat(spec_like(fields))}"
      elif _is_attrs(obj):
        return f"{type(obj)}\n{pprint.pformat(spec_like(_attrs_fields(obj)))}"
      return type(obj)

    def _shape(obj):
      try:
        return list(np.asarray(obj).shape)
      except ValueError:
        return "[unknown shape]"

    return tree_util.tree_map(
        lambda x: f"{_type(x)}{_shape(x)}",
        structure,
        is_leaf=_is_leaf,
    )


except ImportError:
  import tree  # pylint: disable=g-import-not-at-top

  map_structure = tree.map_structure
  map_structure_with_path = tree.map_structure_with_path
  assert_same_structure = tree.assert_same_structure
  flatten = tree.flatten
  flatten_with_path = tree.flatten_with_path
  unflatten_as = tree.unflatten_as

  def spec_like(structure):
    """Infers specification of a tree structure.

    Args:
      structure: The structure to get the spec of.

    Returns:
      Same structure but with the leaves replaced with their stringified spec.
    """

    def _type(obj):
      if isinstance(obj, np.ndarray):
        return np.asarray(obj).dtype
      return type(obj)

    return tree.map_structure(
        lambda x: f"{_type(x)}{list(np.asarray(x).shape)}",
        structure,
    )
