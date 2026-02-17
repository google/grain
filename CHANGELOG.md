# Change log

The change log file hosting all releases with lists of new features and breaking
changes. Best viewed [here](https://google-grain.readthedocs.io/en/latest/changelog.html).

## Unreleased

* New features:
  * Adds support for filtering Grain-internal stack frames from user-thrown
    errors.
  * Adds experimental support for `get_next_index` and `set_next_index` to fetch
    and advance a `grain.DatasetIterator` to the given produced element index.
  * Switches to multithreading instead of multiprocessing in
    `IterDataset.mp_prefetch` when free-threaded Python is detected.
  * `grain.DataLoaderIterator` can now asynchronously start processing elements
    in background with `start_prefetch` call.

* Breaking changes:
  * Custom implementations of `RandomAccessDataSource` should accept `int`
    index in `__getitem__`. Legacy paths that handle `SupportsIndex` will still
    work at runtime, but depending on the type checker in use, if you're
    directly inheriting from `grain.RandomAccessDataSource` and call
    `super().__getitem__` with `Supportsindex` you may see a type checking
    error. Switch to `int` to fix it.

* Deprecations:
  * Deprecates `grain.python.experimental.MultiprocessPrefetchIterDataset`,
    use the graduated version instead: `grain.IterDataset.mp_prefetch`.
  * Deprecates `grain.python.experimental.ConcatenateMapDataset`, use the
    graduated version instead: `grain.MapDataset.concatenate`.

* Bug fixes:
  * Fixes bug in `WindowShuffleIterDataset` checkpointing.

## Grain 0.2.15 (November 25, 2025)

* New features:
  * Adds public `DatasetIterator.close` API as an explicit blocking alternative
    to a cleanup during GC.
  * `DatasetIterator.start_prefetch` now propagates to the first asynchronous
    parent iterator instead of raising `NotImplemented`. This API is useful for
    hiding first batch processing behind model checkpoint recovery.
  * Introduces `grain.experimental.multithread_prefetch` as an
    alternative to multiprocessing prefetch in free-threading Python.
  * Adds experimental support for static `{Map|Iter}Dataset` element
    specification inference.
  * Adds support for changing `IterDataset.mix` components and weights after a
    checkpoint.

## Grain 0.2.14 (October 30, 2025)

* New features
  * Adds Python 3.14 build.
  * Replaces `dm-tree` dependency with pure Python implementation for Pytree
    manipulation. Note that it's only used if `jax` is not installed. If `jax`
    can be imported -- uses `jax.tree_util` instead.

* Breaking changes:
  * Removes `grain[testing]` PyPi build. It is an implementation detail and
    should not be publicly visible.
  * Upgrades linux wheels to `manylinux_2_28`.

* Deprecations:
  * Deprecates Python 3.10 support.
  * Deprecates `grain.python.experimental.visualize_dataset`. Use visualization
    mode instead.

## Grain 0.2.13 (October 15, 2025)

* New features
  * Adds `reseed_each_epoch` option to `MapDataset.repeat` that allows to replay
    the first epoch exactly if set to False (True by default).
  * Introduces `grain.experimental.RebatchIterDataset` for efficient rebatch.
  * Migrates data loader to use dataset API under the hood.
  * Improves first-fit packing speed by up to 12x.
  * Adds best-fit packing implementation which reduces padding in benchmarks by
    over 27% compared to first-fit.
  * Adds `max_sequences_per_bin` to packing transformations to limit the number
    of sequences packed into a single bin.
  * Introduces `grain.experimental.RepeatIterDataset`.
  * Adds custom batching function support to `grain.DataLoader`.
  * Adds `grain.experimental.FlatMapTransform` support to `grain.DataLoader`.
  * Introduces `grain.experimental.CacheIterDataset` for caching parent dataset.

* Breaking changes:
  * SliceMapDataset updated to use the full index relative to the parent
    dataset, instead index%len(self).

* Deprecations:
  * Graduates `grain.experimental.apply_transformations` to
   `grain.{MapDataset|IterDataset}.apply`. The experimental API will soon be
    deprecated.

* Bug fixes
  * Fixes memory leak on `ThreadPrefetchDatasetIterator` deletion.

## Grain 0.2.12 (August 21, 2025)

* New features:
  * Adds Windows build.
  * Allow passing `read_kwargs` to `ParquetIterDataset` for configuring parquet
    file reading.
  * `ThreadPrefetchDatasetIterator` now supports non-Grain iterators that
    support checkpointing.
  * Introduces API for device prefetch - `grain.experimental.device_put()` for
    easy CPU and device prefetching.
  * Introduces API for autotuning -- given the user provided RAM restrictions
    and specific `IterDataset`, finds number of processes for `mp_prefetch`
    and buffer size for `PrefetchDatasetIterator`.
  * Allow passing `reader_options` to `ArrayRecordDataSource` for configuring
    array record file reading.
  * Introduces `grain.experimental.batch_and_pad` for padding a partial batch to
    avoid dropping batch remainder data.
  * Grain interleave optimization - allow creating more threads to parallelly
    keep starting iterators and prefetching elements.
  * Allow for alternative slicing of the data for
    `MultiprocessPrefetchIterDataset`. New slicing allows each worker process to
    read unique file shards and thus improving performance.

* Breaking changes:
  * Upgrades `array_record` and `protobuf`.

* Deprecations:

* Bug fixes

## Grain 0.2.11 (July 2, 2025)

* New features:
  * Automatic publishing releases to PyPI via GitHub actions.
  * Nightly builds.
  * Introduced changelog.
