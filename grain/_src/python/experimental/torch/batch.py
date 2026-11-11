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


class _TorchDefaultCollate:
  """Picklable lazy wrapper around ``torch.utils.data.default_collate``.

  Implements ``batch.BatchFn`` (``__call__`` + ``output_spec``) so
  ``grain.experimental.get_element_spec`` continues to work on batched
  ``grain.experimental.torch`` datasets without importing ``torch`` eagerly.
  """

  def __call__(self, elements: Sequence[Any]) -> Any:
    from torch.utils.data import default_collate as torch_default_collate  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    return torch_default_collate(list(elements))

  def output_spec(
      self, input_spec: Any, batch_size: int, drop_remainder: bool
  ) -> Any:
    dim0 = batch_size if drop_remainder else None
    return tree_lib.map_structure(
        lambda x: base.ShapeDtypeStruct(shape=(dim0,) + x.shape, dtype=x.dtype),
        input_spec,
    )


default_collate = _TorchDefaultCollate()
