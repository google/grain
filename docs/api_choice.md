# Choice of API





Grain offers two different ways of defining data processing pipelines:
[`DataLoader`](tutorials/data_loader_tutorial.md) and [`Dataset`](tutorials/dataset_basic_tutorial.md).

> TL;DR: If you need to do one of the following:
>
> *   mix multiple data sources
> *   pack variable length elements
> *   split dataset elements and globally shuffle the splits
>
> then you should use `Dataset`, otherwise use simpler `DataLoader`.

## `DataLoader`

`DataLoader` is a high-level API that uses the following abstractions to define
data processing:

*   [`RandomAccessDataSource`](https://github.com/google/grain/tree/main/grain/_src/python/data_sources.py)
    that reads raw input data.
*   A
    [`Sampler`](https://github.com/google/grain/tree/main/grain/_src/python/samplers.py)
    that defines the order in which the raw data should be read.
*   A flat sequence of
    [`Transformation`s](https://github.com/google/grain/tree/main/grain/_src/core/transforms.py)
    to apply to the raw data.

You can specify other execution parameters for asynchronous data processing,
sharding, shuffling, and `DataLoader` will automatically take care of inserting
them in the right places between the data processing steps.

These are simple and usually general enough to cover most data processing use
cases. Prefer using `DataLoader` if your workflow can be described using the
abstractions above. See [tutorial](tutorials/data_loader_tutorial.md)
for more details.

## `Dataset`

`Dataset` is a lower-level API that uses chaining syntax to define data
transformation steps. It allows more general types of processing (e.g. dataset
mixing) and more control over the execution (e.g. different order of data
sharding and shuffling). `Dataset` transformations are composed in a way that
allows to preserve random access property past the source and some of the
transformations. This, among other things, can be used for debugging by
evaluating dataset elements at specific positions without processing the entire
dataset.

There are 3 main classes comprising the `Dataset` API:

*   [`MapDataset`](https://github.com/google/grain/tree/main/grain/_src/python/dataset/dataset.py)
    defines a dataset that supports efficient random access. Think of it as an
    (infinite) `Sequence` that computes values lazily.
*   [`IterDataset`](https://github.com/google/grain/tree/main/grain/_src/python/dataset/dataset.py)
    defines a dataset that does not support efficient random access and only
    supports iterating over it. It's an `Iterable`. Any `MapDataset` can be
    turned into a `IterDataset` by calling `to_iter_dataset()`.
*   [`DatasetIterator`](https://github.com/google/grain/tree/main/grain/_src/python/dataset/dataset.py)
    defines a stateful iterator of an `IterDataset`. The state of the iterator
    can be saved and restored.

Most data pipelines will start with one or more `MapDataset` (often derived from
a `RandomAccessDataSource`) and switch to `IterDataset` late or not at all. See
[tutorial](tutorials/dataset_basic_tutorial.md)
for more details.
