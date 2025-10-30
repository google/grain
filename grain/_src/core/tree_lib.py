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

import collections
import dataclasses
import itertools
import pprint

from etils import epy
import numpy as np

_SEQUENCE_TYPES = (list, tuple)
_MAPPING_TYPES = (dict, collections.abc.Mapping)
_ALL_TYPES = _SEQUENCE_TYPES + _MAPPING_TYPES

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

except ImportError:

  def map_structure(f, *trees, is_leaf=None):
    """Pure Python version of `jax.tree_util.tree_map`."""
    a = trees[0]
    if is_leaf is not None and is_leaf(a):
      return f(*trees)
    elif isinstance(a, _SEQUENCE_TYPES):
      new_items = (map_structure(f, *v, is_leaf=is_leaf) for v in zip(*trees))
      if epy.is_namedtuple(a):
        return type(a)(*new_items)
      else:
        return type(a)(new_items)
    elif isinstance(a, _MAPPING_TYPES):
      new_items = (
          (k, map_structure(f, *v, is_leaf=is_leaf))
          for k, v in epy.zip_dict(*trees)
      )
      if isinstance(a, collections.defaultdict):
        new_tree = type(a)(a.default_factory)
        new_tree.update(new_items)
        return new_tree
      else:
        return type(a)(new_items)
    else:  # leaf
      return f(*trees)

  def flatten(structure):
    if isinstance(structure, _SEQUENCE_TYPES):
      return list(itertools.chain.from_iterable(flatten(v) for v in structure))
    elif isinstance(structure, _MAPPING_TYPES):
      return list(
          itertools.chain.from_iterable(
              flatten(v) for _, v in structure.items()
          )
      )
    else:  # leaf
      return [structure]

  def unflatten_as(structure, flat_sequence):
    return _unflatten_iter_as(structure, iter(flat_sequence))

  def _unflatten_iter_as(structure, flat_iter):
    """`unflatten` recursive implementation."""
    if isinstance(structure, _SEQUENCE_TYPES):
      new_items = (_unflatten_iter_as(v, flat_iter) for v in structure)
      if epy.is_namedtuple(structure):
        return type(structure)(*new_items)
      else:
        return type(structure)(new_items)
    elif isinstance(structure, _MAPPING_TYPES):
      new_items = (
          (k, _unflatten_iter_as(v, flat_iter)) for k, v in structure.items()
      )
      if isinstance(structure, collections.defaultdict):
        new_tree = type(structure)(structure.default_factory)
        new_tree.update(new_items)
        return new_tree
      else:
        return type(structure)(new_items)
    else:  # leaf
      return next(flat_iter)

  def assert_same_structure(a, b):
    try:
      _assert_same_structure(a, b)
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix="The two structures don't match: ")

  def _assert_same_structure(a, b):
    """`assert_same_structure` recursive implementation."""
    if isinstance(a, _ALL_TYPES):
      if not isinstance(a, type(b)):
        raise AssertionError(
            f"structures have different types: {type(a)} != {type(b)}"
        )
    if isinstance(a, _SEQUENCE_TYPES):
      if len(a) != len(b):
        raise AssertionError(
            f"sequences have different lengths: {len(a)} != {len(b)}"
        )
      for i, (v0, v1) in enumerate(zip(a, b)):
        try:
          _assert_same_structure(v0, v1)
        except AssertionError as e:
          epy.reraise(e, prefix=f"In {i}: ")
    elif isinstance(a, _MAPPING_TYPES):
      k0 = sorted(a)
      k1 = sorted(b)
      if k0 != k1:
        raise AssertionError(f"dict keys do not match: {k0} != {k1}")
      # Flatten sort the keys, so reconstruct the ordered sorted
      for k, (v0, v1) in epy.zip_dict(a, b):
        try:
          _assert_same_structure(v0, v1)
        except AssertionError as e:
          epy.reraise(e, prefix=f"In {k}: ")
    else:  # leaf
      return

  def flatten_with_path(structure):
    return _flatten_with_path((), structure)

  def _flatten_with_path(prefix, structure):
    """Recursive implementation of `flatten_with_path`."""
    if isinstance(structure, _SEQUENCE_TYPES):
      return list(
          itertools.chain.from_iterable(
              _flatten_with_path(prefix + (i,), v)
              for i, v in enumerate(structure)
          )
      )
    elif isinstance(structure, _MAPPING_TYPES):
      return list(
          itertools.chain.from_iterable(
              _flatten_with_path(prefix + (k,), v) for k, v in structure.items()
          )
      )
    else:  # leaf
      return [(prefix, structure)]

  def map_structure_with_path(f, *trees):
    return _map_structure_with_path((), f, *trees)

  def _map_structure_with_path(prefix, f, *trees, is_leaf=None):
    """Recursive implementation of `map_structure_with_path`."""
    a = trees[0]
    if is_leaf is not None and is_leaf(a):
      return f(prefix, *trees)
    elif isinstance(a, _SEQUENCE_TYPES):
      new_items = (
          _map_structure_with_path(prefix + (i,), f, *v, is_leaf=is_leaf)
          for i, v in enumerate(zip(*trees))
      )
      if epy.is_namedtuple(a):
        return type(a)(*new_items)
      else:
        return type(a)(new_items)
    elif isinstance(a, _MAPPING_TYPES):
      new_items = (
          (k, _map_structure_with_path(prefix + (k,), f, *v, is_leaf=is_leaf))
          for k, v in epy.zip_dict(*trees)
      )
      if isinstance(a, collections.defaultdict):
        new_tree = type(a)(a.default_factory)
        new_tree.update(new_items)
        return new_tree
      else:
        return type(a)(new_items)
    else:  # leaf
      return f(prefix, *trees)


def map_structure_up_to(shallow_structure, f, structure):
  """Applies `f` to the `structure` up to the given `shallow_structure`."""
  if isinstance(shallow_structure, _SEQUENCE_TYPES):
    new_items = (
        map_structure_up_to(a, f, b)
        for a, b in zip(shallow_structure, structure, strict=False)
    )
    if epy.is_namedtuple(shallow_structure):
      return type(shallow_structure)(*new_items)
    else:
      return type(shallow_structure)(new_items)
  elif isinstance(shallow_structure, _MAPPING_TYPES):
    new_items = {
        k: map_structure_up_to(v, f, structure[k])
        for k, v in shallow_structure.items()
    }
    if isinstance(shallow_structure, collections.defaultdict):
      new_tree = type(shallow_structure)(shallow_structure.default_factory)
      new_tree.update(new_items)
      return new_tree
    else:
      return type(shallow_structure)(new_items)
  else:  # leaf
    return f(structure)


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
        inner_type = type(obj[0]).__name__
      else:
        inner_type = ""
      return f"{container_type}<{inner_type}>"
    elif dataclasses.is_dataclass(obj):
      # We avoid registering the dataclass with jax"s tree_util because
      # that would affect the behavior outside of this module. We also avoid
      # using dataclasses.asdict() since it makes data copy.
      fields = {}
      for field in dataclasses.fields(obj):
        fields[field.name] = getattr(obj, field.name)
      return f"{type(obj).__name__}\n{pprint.pformat(spec_like(fields))}"
    elif _is_attrs(obj):
      return f"{type(obj).__name__}\n{pprint.pformat(spec_like(_attrs_fields(obj)))}"
    return type(obj).__name__

  def _shape(obj):
    try:
      return list(np.asarray(obj).shape)
    except ValueError:
      return "[unknown shape]"

  return map_structure(
      lambda x: f"{_type(x)}{_shape(x)}",
      structure,
      is_leaf=_is_leaf,
  )
