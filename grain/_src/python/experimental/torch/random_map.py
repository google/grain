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
"""Experimental random map transformation that accepts a PyTorch PRNG key.

PyTorch is an optional dependency: it is imported lazily on first element
access, so importing or constructing ``TorchRandomMapWithKey`` does not
require PyTorch.
"""

from collections.abc import Callable
import functools
from typing import Any

from grain._src.core import transforms


class TorchRandomMapWithKey:
  """``RandomMapWithKey`` that passes a PyTorch PRNG key to the transformation.

  Can be used in two ways:

  1. Wrapping a callable ``fn(element, key)``: ``TorchRandomMapWithKey(fn)``
     (used automatically when passing a plain callable to
     ``grain.experimental.torch.MapDataset.random_map_with_key(fn)`` or
     ``IterDataset.random_map_with_key(fn)``).
  2. Subclassing and overriding ``random_map_with_key(self, element, key)``.

  Each element receives a Philox PRNG key: a CPU ``uint64[2]`` tensor holding a
  pair of integer seeds, derived via ``torch.func._random.key(seed)`` and
  ``torch.func._random.fold_in(op_key, index)``. Pass ``key`` to the random
  operations in ``torch.func._random`` (``normal``, ``uniform``, ``randint``,
  ``bits``) and use ``torch.func._random.split(key)`` when drawing multiple
  independent random values within one transform. The operations do not modify
  ``key``, so the same key always gives the same random values.

  Each element's key depends only on the dataset seed and the element's index,
  so results are deterministic under random access, prefetch threads, worker
  processes, and checkpoint restores. The global PyTorch RNG is neither used nor
  modified.

  Experimental: both this class and PyTorch's ``torch.func._random`` key-based
  PRNG module are experimental, and their APIs or random streams may change as
  PyTorch stabilizes public key-based RNG support.

  Example:
    Adding Gaussian noise drawn from each element's PRNG key::

      import torch
      import torch.func._random as torch_random
      from grain.experimental import torch as grain_torch

      def add_noise(x, key):
        return x + torch_random.normal(key, x.shape)

      # `grain.experimental.torch` datasets wrap plain callables in
      # `TorchRandomMapWithKey` automatically.
      ds = (
          grain_torch.MapDataset.source([torch.zeros(2)] * 4)
          .random_map_with_key(add_noise, seed=42)
          .batch(2)
      )

      # The same index always produces the same random values.
      print(torch.equal(ds[1], ds[1]))
      # True

    ``grain.MapDataset`` and ``grain.IterDataset`` wrap plain callables in
    ``JaxRandomMapWithKey``, so wrap the callable explicitly there::

      import grain

      ds = grain.MapDataset.source([torch.zeros(2)] * 4).random_map_with_key(
          grain_torch.TorchRandomMapWithKey(add_noise), seed=42
      )
  """

  def __init__(self, fn: Callable[[Any, Any], Any] | None = None):
    if fn is not None:
      if isinstance(fn, transforms.RandomMap) or not callable(fn):
        raise TypeError(
            "`TorchRandomMapWithKey` expects a callable"
            f" `transform(element, key)`, got {type(fn).__name__}."
        )
    self._wrapped_fn = fn

  def make_key_for_index(self, seed: int) -> Callable[[int], Any]:
    """Returns ``index -> key`` closure derived lazily via ``torch.func._random``."""
    # May raise ImportError if PyTorch is not linked or lacks key-based RNG.
    from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    return functools.partial(torch_random.fold_in, torch_random.key(seed))

  def random_map_with_key(self, element: Any, key: Any) -> Any:
    """Applies the wrapped callable or subclass override with ``key``."""
    if self._wrapped_fn is None:
      raise NotImplementedError(
          "Subclasses of `TorchRandomMapWithKey` must override"
          " `random_map_with_key` or pass `fn` to `__init__`."
      )
    return self._wrapped_fn(element, key)


def as_torch_random_map_with_key(
    transform: transforms.RandomMapWithKey | Callable[[Any, Any], Any],
    owner_name: str,
) -> transforms.RandomMapWithKey:
  """Normalizes a `RandomMapWithKey` or callable into `TorchRandomMapWithKey`."""
  if isinstance(transform, transforms.RandomMapWithKey):
    return transform
  if isinstance(transform, transforms.RandomMap) or not callable(transform):
    raise TypeError(
        f"`{owner_name}` expects a callable `transform(element, key)` or"
        f" `RandomMapWithKey`, got {type(transform).__name__}."
    )
  return TorchRandomMapWithKey(transform)
