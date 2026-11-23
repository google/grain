# PyTorch support for Grain

This directory implements
[`grain.experimental.torch`](/grain/experimental/torch.py), which
provides Grain's Dataset API with PyTorch defaults.

The module is experimental and its API may change. It relies on PyTorch's
`torch.func._random` module, which is experimental as well.

## Usage

The modules in this directory are private. Import the public module instead,
under an alias so that it does not shadow `torch`:

```python
import torch
import torch.func._random as torch_random
from grain.experimental import torch as grain_torch


def add_noise(x, key):  # `key` is a PyTorch PRNG key.
  return x + torch_random.normal(key, x.shape)


ds = (
    grain_torch.MapDataset.source([torch.zeros(1024)] * 8)
    .seed(42)
    .shuffle()
    .random_map_with_key(add_noise)
    .batch(4)  # Collates with `torch.utils.data.default_collate`.
    .to_iter_dataset()
    .mp_prefetch(grain_torch.MultiprocessingOptions(num_workers=2))
)

it = iter(ds)
next(it).shape  # torch.Size([4, 1024])
it.close()
```

The docstring of the public module has a complete example of a training loop,
with sharding across `torch.distributed` ranks and checkpointing.

## Differences from plain Grain

`MapDataset`, `IterDataset` and `DatasetIterator` support the same methods as
`grain.MapDataset`, `grain.IterDataset` and `grain.DatasetIterator`, which this
document calls plain Grain. They behave the same except for the following:

*   `random_map_with_key` wraps a plain callable `fn(element, key)` in
    `TorchRandomMapWithKey` instead of `JaxRandomMapWithKey`, so `key` is a
    PyTorch PRNG key.
*   `random_map` and `RandomMap` transformations are not supported, because they
    pass a NumPy random number generator to the transformation. Use
    `random_map_with_key` instead.
*   `batch` collates with `torch.utils.data.default_collate` by default, so
    batches are `torch.Tensor`s.
*   `mp_prefetch` limits each worker process to
    `MultiprocessingOptions.worker_intra_op_threads` PyTorch intra-op threads (1
    by default). It also passes CPU tensors between processes through Grain's
    shared memory, so `DatasetOptions.min_shm_size` applies to them.
*   `IterDataset` has a `with_options` method, the method form of
    `grain.experimental.WithOptionsIterDataset`.
*   `MapDataset.from_dataset` and `IterDataset.from_dataset` give the PyTorch
    defaults to a pipeline that was built with plain Grain.
*   Iterators implement `state_dict` and `load_state_dict`, so they can be
    passed to `save` and `load` of `torch.distributed.checkpoint` next to the
    model and the optimizer. The state is stored per rank.

## Files

The public module re-exports the public names of these modules:

*   `dataset.py`: `MapDataset`, `IterDataset`, `DatasetIterator` and
    `mp_prefetch`. The classes are facades over the plain Grain classes that
    supply the PyTorch defaults.
*   `random_map.py`: `TorchRandomMapWithKey`, a `RandomMapWithKey` that
    derives a PyTorch PRNG key for each element.
*   `options.py`: `MultiprocessingOptions`, which is
    `grain.MultiprocessingOptions` with `worker_intra_op_threads`.
*   `checkpoint.py`: `IteratorCheckpointWrapper`, which saves and restores the
    iterator state of each rank with `torch.distributed.checkpoint`.
*   `distributed.py`: `get_process_index_and_count`, which returns the rank of
    the current process and the number of ranks.

The public module also exports `DatasetOptions`, which is the same class as
`grain.experimental.DatasetOptions`.

These modules have no public names:

*   `batch.py`: the default `batch_fn`, which is
    `torch.utils.data.default_collate` with element spec inference.
*   `prefetch.py`: the implementation of `mp_prefetch`.

## Implementation notes

*   **Facades.** `MapDataset` and `IterDataset` hold a plain Grain dataset in
    `_inner`, build every transformation on it, and wrap the result. A pipeline
    is therefore made of plain Grain datasets only, with a single facade on the
    outside. To keep it that way, unwrap a dataset before handing it to plain
    Grain, so that a facade never ends up inside a pipeline or inside another
    facade.
*   **Optional dependency.** Importing `grain` or `grain.experimental.torch`
    does not import `torch`. The modules import it inside the functions that
    need it, never at module level, and the libraries in `BUILD` do not depend
    on it.
*   **Tensors in shared memory.** Grain's `mp_prefetch` moves NumPy arrays from
    the workers through shared memory and pickles everything else. In a worker,
    `prefetch.py` therefore replaces each CPU tensor with the NumPy array that
    shares its memory, and it rebuilds the tensor from the array in the main
    process. A tensor that cannot be passed this way, such as one that requires
    gradients, is pickled.
*   **Checkpoints.** When it saves, `torch.distributed.checkpoint` keeps a
    single copy of each non-tensor value of a state dictionary across ranks.
    `IteratorCheckpointWrapper` therefore stores the state of each rank under a
    key of its own, and checks the number of ranks when it loads the state.
    Saving from more than one rank requires PyTorch 2.4 or later.

## Tests

Each `*_test.py` file tests the module of the same name. `dataset_test.py` also
covers `batch.py`, and `prefetch_test.py` covers `options.py`.
