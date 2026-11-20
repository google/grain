# Copyright 2026 Google LLC
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
"""Default PyTorch batch collation with Grain element-spec inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from grain._src.core import tree_lib
from grain._src.python.dataset import base


def _is_plain_tuple(x: Any) -> bool:
  return isinstance(x, tuple) and not hasattr(x, "_fields")


def _batched_spec(spec: Any, dim0: int | None) -> Any:
  """Returns the spec of what `default_collate` returns for `spec` elements.

  Args:
    spec: The spec of the elements.
    dim0: The size of the batch dimension, or `None` if it is not known.
  """

  def batch_leaf(x: Any) -> Any:
    if _is_plain_tuple(x):
      # `default_collate` returns a list for a tuple that is not a named tuple.
      return [_batched_spec(item, dim0) for item in x]
    return base.ShapeDtypeStruct(shape=(dim0,) + x.shape, dtype=x.dtype)

  # TODO: Traverse the spec with a PyTorch backend of `tree_lib`
  # once it exists, so that types registered with `torch.utils._pytree` are
  # traversed the same way as in PyTorch.
  return tree_lib.map_structure(batch_leaf, spec, is_leaf=_is_plain_tuple)


class _TorchDefaultCollate:
  """Picklable lazy wrapper around ``torch.utils.data.default_collate``.

  Implements ``batch.BatchFn`` (``__call__`` + ``output_spec``) so
  ``grain.experimental.get_element_spec`` continues to work on batched
  ``grain.experimental.torch`` datasets without importing ``torch`` eagerly.
  """

  def __call__(self, elements: Sequence[Any]) -> Any:
    from torch.utils.data import default_collate as torch_default_collate  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    return torch_default_collate(elements)

  def output_spec(
      self, input_spec: Any, batch_size: int, drop_remainder: bool
  ) -> Any:
    dim0 = batch_size if drop_remainder else None
    return _batched_spec(input_spec, dim0)


default_collate = _TorchDefaultCollate()
