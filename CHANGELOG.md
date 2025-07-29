# Change log

The change log file hosting all releases with lists of new features and breaking
changes. Best viewed [here](https://google-grain.readthedocs.io/en/latest/changelog.html).

## Unreleased

* New features:
  * Allow passing `read_kwargs` to `ParquetIterDataset` for configuring parquet
    file reading.
  * `ThreadPrefetchDatasetIterator` now supports non-Grain iterators that
    support checkpointing.
  * Introduces API for device prefetch - `grain.experimental.device_put()` for
    easy CPU and device prefetching.
  * Adds Windows build.
  * Introduces API for autotuning -- finds number of processes for
    `mp_prefetch`, given the user provided RAM restrictions
    and specific `IterDataset`.
  * Allow passing `reader_options` to `ArrayRecordDataSource` for configuring
    array record file reading.

* Breaking changes:

* Deprecations:

* Bug fixes

## Grain 0.2.11 (July 2, 2025)

* New features:
  * Automatic publishing releases to PyPI via GitHub actions.
  * Nightly builds.
  * Introduced changelog.