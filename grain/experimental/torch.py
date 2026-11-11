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
"""Experimental Grain APIs for PyTorch pipelines.

``grain.experimental.torch`` provides Grain's Dataset API with PyTorch defaults:

- Random transformations receive PyTorch PRNG keys
  (``stateless_random_map``).
- ``batch`` collates elements into ``torch.Tensor``s.
- ``mp_prefetch`` limits PyTorch threads in worker processes and passes CPU
  tensors between processes through shared memory.
- Iterators can be checkpointed with ``torch.distributed.checkpoint``.

PyTorch remains an optional dependency: importing ``grain`` or
``grain.experimental.torch`` does not import ``torch``.

Import the module under an alias so that it does not shadow ``torch``::

  from grain.experimental import torch as grain_torch

Example usage::

  import torch
  import torch.distributed.checkpoint as dcp
  import torch.func._random as torch_random
  from grain.experimental import torch as grain_torch

  def augment(sample, key):
    image, label = sample
    noise = torch_random.normal(key, image.shape)
    return image + 0.1 * noise, label

  # The rank and the number of ranks come from `torch.distributed`; they are
  # `(0, 1)` outside of a distributed job.
  rank, world_size = grain_torch.get_process_index_and_count()

  ds = (
      # Any indexable source, such as a map-style `torch.utils.data.Dataset`.
      grain_torch.MapDataset.source(source)
      .seed(42)
      .shuffle()
      .repeat(num_epochs)
      [rank::world_size]  # Each rank reads its own part of the data.
      .stateless_random_map(augment)  # `key` is a PyTorch PRNG key.
      .batch(256, drop_remainder=True)  # `torch.utils.data.default_collate`.
      .to_iter_dataset()
      .with_options(grain_torch.DatasetOptions(min_shm_size=64 << 10))
      # Each worker process uses one PyTorch intra-op thread by default.
      .mp_prefetch(grain_torch.MultiprocessingOptions(num_workers=16))
  )

  it = iter(ds)
  for step, (images, labels) in enumerate(it):
    ...  # Run a training step.
    if step % checkpoint_every == 0:
      dcp.save({"model": model, "data": it}, checkpoint_id=checkpoint_path)

  # To resume, rebuild `ds` and restore the position of a new iterator.
  it = iter(ds)
  dcp.load({"model": model, "data": it}, checkpoint_id=checkpoint_path)

The datasets support the same methods as ``grain.MapDataset`` and
``grain.IterDataset``; see their documentation for the transformations that are
not specific to PyTorch.

This module is experimental and its API may change. It relies on PyTorch's
``torch.func._random`` module, which is experimental as well.
"""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-multiple-import
# pylint: disable=unused-import

from grain._src.python.dataset.base import (
    DatasetOptions,
)
from grain._src.python.experimental.torch.checkpoint import (
    IteratorState,
)
from grain._src.python.experimental.torch.dataset import (
    DatasetIterator,
    IterDataset,
    MapDataset,
)
from grain._src.python.experimental.torch.distributed import (
    get_process_index_and_count,
)
from grain._src.python.experimental.torch.options import (
    MultiprocessingOptions,
)
from grain._src.python.experimental.torch.prefetch import (
    mp_prefetch,
)
from grain._src.python.experimental.torch.random_map import (
    TorchStatelessRandomMap,
)
