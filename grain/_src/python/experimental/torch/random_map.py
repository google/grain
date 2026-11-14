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
"""Experimental stateless PyTorch PRNG random map transformation.

PyTorch is an optional dependency: it is imported lazily on first element
access, so importing or constructing ``TorchStatelessRandomMap`` does not
require PyTorch.
"""

from collections.abc import Callable
import functools
from typing import Any

from grain._src.core import transforms


class TorchStatelessRandomMap:
  """Stateless PyTorch random map implementing ``StatelessRandomMap``.

  Can be used in two ways:

  1. Wrapping a callable ``fn(element, key)``: ``TorchStatelessRandomMap(fn)``
     (used automatically when passing a plain callable to
     ``grain.experimental.torch.MapDataset.stateless_random_map(fn)`` or
     ``IterDataset.stateless_random_map(fn)``).
  2. Subclassing and overriding ``stateless_random_map(self, element, key)``.

  Each element receives an immutable CPU ``uint64[2]`` Philox PRNG key tensor
  derived via ``torch.func._random.key(seed)`` and
  ``torch.func._random.fold_in(op_key, index)``. Pass ``key`` to stateless
  random operations in ``torch.func._random`` (``normal``, ``uniform``,
  ``randint``, ``bits``) and use ``torch.func._random.split(key)`` when drawing
  multiple independent random values within one transform.

  Each element's key depends only on the dataset seed and the element's index,
  so results are deterministic under random access, prefetch threads, worker
  processes, and checkpoint restores. The global PyTorch RNG is neither used nor
  modified.

  Experimental: both this class and PyTorch's ``torch.func._random`` stateless
  PRNG module are experimental, and their APIs or random streams may change as
  PyTorch stabilizes public stateless RNG support.

  Example:
    Adding Gaussian noise drawn from each element's stateless PRNG key::

      import grain
      import torch
      import torch.func._random as torch_random

      def add_noise(x, key):
        return x + torch_random.normal(key, x.shape)

      ds = (
          grain.MapDataset.source([torch.zeros(2)] * 4)
          .stateless_random_map(
              grain.experimental.torch.TorchStatelessRandomMap(add_noise),
              seed=42,
          )
          .batch(2, batch_fn=torch.utils.data.default_collate)
      )

      # The same index always produces the same random values.
      print(torch.equal(ds[1], ds[1]))
      # True
  """

  def __init__(self, fn: Callable[[Any, Any], Any] | None = None):
    if fn is not None:
      if isinstance(fn, transforms.RandomMap) or not callable(fn):
        raise TypeError(
            "`TorchStatelessRandomMap` expects a callable"
            f" `transform(element, key)`, got {type(fn).__name__}."
        )
    self._wrapped_fn = fn

  def make_key_for_index(self, seed: int) -> Callable[[int], Any]:
    """Returns ``index -> key`` closure derived lazily via ``torch.func._random``."""
    # May raise ImportError if PyTorch is not linked or lacks stateless RNG.
    from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    return functools.partial(torch_random.fold_in, torch_random.key(seed))

  def stateless_random_map(self, element: Any, key: Any) -> Any:
    """Applies the wrapped callable or subclass override with ``key``."""
    if self._wrapped_fn is None:
      raise NotImplementedError(
          "Subclasses of `TorchStatelessRandomMap` must override"
          " `stateless_random_map` or pass `fn` to `__init__`."
      )
    return self._wrapped_fn(element, key)


def as_torch_stateless_random_map(
    transform: transforms.StatelessRandomMap | Callable[[Any, Any], Any],
    owner_name: str,
) -> transforms.StatelessRandomMap:
  """Normalizes a `StatelessRandomMap` or callable into `TorchStatelessRandomMap`."""
  if isinstance(transform, transforms.StatelessRandomMap):
    return transform
  if isinstance(transform, transforms.RandomMap) or not callable(transform):
    raise TypeError(
        f"`{owner_name}` expects a callable `transform(element, key)` or"
        f" `StatelessRandomMap`, got {type(transform).__name__}."
    )
  return TorchStatelessRandomMap(transform)
