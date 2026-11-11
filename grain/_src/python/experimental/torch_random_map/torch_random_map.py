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
"""Experimental random map that hands each element a stateless PyTorch PRNG key.

PyTorch is an optional dependency: it is imported lazily on first element
access, so importing or constructing this dataset does not require PyTorch.
"""

import functools
from typing import Any, Callable, Sequence, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import map as map_dataset

T = TypeVar("T")


class TorchRandomMapMapDataset(
    map_dataset._ElementSpecFromTransformMapDatasetMixin[T]  # pylint: disable=protected-access
):
  """Random map transformation for ``MapDataset`` using stateless PyTorch PRNG.

  Like ``MapDataset.random_map``, but instead of a stateful
  ``np.random.Generator``, the transform receives an immutable per-element
  PyTorch Philox PRNG ``key`` tensor derived via ``torch.func._random.key`` and
  ``torch.func._random.fold_in``. Pass ``key`` to stateless random operations in
  ``torch.func._random`` (``normal``, ``uniform``, ``randint``, ``bits``) and
  use ``torch.func._random.split(key)`` when drawing multiple independent random
  values within one transform.

  Each element's key depends only on the dataset seed and the element's index,
  so results are deterministic under random access, prefetch threads, worker
  processes, and checkpoint restores. The global PyTorch RNG is neither used nor
  modified.

  The seed defaults to the one derived from ``ds.seed(...)``, as for
  ``random_map``.

  Experimental: both this class and PyTorch's ``torch.func._random`` stateless
  PRNG module are experimental, and their APIs or random streams may change as
  PyTorch stabilizes public stateless RNG support. Use with ``ds.pipe(...)``;
  ``ds.apply(...)`` dispatch does not support it.

  Example:
    Adding Gaussian noise drawn from each element's stateless PRNG key::

      import grain
      import torch
      import torch.func._random as torch_random

      def add_noise(x, key):
        return x + torch_random.normal(key, x.shape)

      ds = (
          grain.MapDataset.source([torch.zeros(2)] * 4)
          .pipe(grain.experimental.TorchRandomMapMapDataset, add_noise, seed=42)
          .batch(2, batch_fn=torch.utils.data.default_collate)
      )

      # The same index always produces the same random values.
      print(torch.equal(ds[1], ds[1]))
      # True

    Splitting an element's key across multiple stateless random operations::

      from torchvision.transforms.v2 import functional as F

      def augment(image, key):
        flip_key, noise_key = torch_random.split(key)
        if torch_random.uniform(flip_key, ()) < 0.5:
          image = F.horizontal_flip(image)
        return image + 0.01 * torch_random.normal(noise_key, image.shape)
  """

  def __init__(
      self,
      parent: dataset.MapDataset,
      transform: Callable[[Any, Any], T],
      seed: int | None = None,
  ):
    """Initializes the TorchRandomMapMapDataset.

    Args:
      parent: The parent dataset.
      transform: A callable ``fn(element, key)`` that receives an element and a
        stateless PyTorch PRNG key tensor derived for the element.
      seed: Non-negative 64-bit integer seed. Defaults to the seed derived from
        ``ds.seed()``.
    """
    if not callable(transform):
      raise TypeError(
          "`TorchRandomMapMapDataset` expects a callable"
          f" `transform(element, key)`, got {type(transform).__name__}."
      )
    super().__init__(parent, transform)
    self._map_fn = transform
    seed = self._default_seed if seed is None else seed
    if seed is None:
      raise ValueError(
          "`TorchRandomMapMapDataset` requires a seed. Please either provide it"
          " with `ds.seed(seed)` before any random transformations or pass it"
          " directly with `TorchRandomMapMapDataset(parent, transform,"
          " seed=seed)`."
      )
    seed = int(seed)
    if not 0 <= seed < 2**64:
      raise ValueError(f"Seed must be in [0, 2**64), got {seed}.")
    self._seed = seed

  def __getstate__(self) -> dict[str, Any]:
    state = self.__dict__.copy()
    state.pop("_key_for_index", None)
    return state

  def __len__(self) -> int:
    return len(self._parent)

  def __str__(self) -> str:
    return f"TorchRandomMapMapDataset(transform={self._transform_name})"

  @functools.cached_property
  def _key_for_index(self) -> Callable[[int], Any]:
    # May raise ImportError if PyTorch is not linked or lacks stateless RNG.
    from torch.func import _random as torch_random  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    return functools.partial(torch_random.fold_in, torch_random.key(self._seed))

  def _random_map_element(self, element: Any, index: int) -> T | None:
    if element is None:
      return None
    return self._map_fn(element, self._key_for_index(index))

  def __getitem__(self, index):
    if isinstance(index, slice):
      return self.slice(index)
    element = self._parent[index]
    with self._stats.record_self_time():
      mapped_element = self._random_map_element(element, index)
      return (
          self._stats.record_output_spec(mapped_element)
          if mapped_element is not None
          else None
      )

  def _getitems(self, indices: Sequence[int]):
    elements = self._parent._getitems(indices)  # pylint: disable=protected-access
    with self._stats.record_self_time(num_elements=len(indices)):
      processed_elements = [
          self._random_map_element(element, index)
          for element, index in zip(elements, indices)
      ]
    return self._stats.record_output_spec_for_batch(processed_elements)
