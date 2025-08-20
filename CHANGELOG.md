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
  * Introduces API for autotuning -- given the user provided RAM restrictions
    and specific `IterDataset`, finds number of processes for `mp_prefetch`
    and buffer size for `PrefetchDatasetIterator`.
  * Allow passing `reader_options` to `ArrayRecordDataSource` for configuring
    array record file reading.
  * Introduces `grain.experimental.batch_and_pad` for padding a partial batch to
    avoid dropping batch remainder data.
  * Upgrades `array_record` and `protobuf`.
  * Make keyfile optional when only doing lookup-by-key with sstable sources

* Breaking changes:

* Deprecations:

* Bug fixes

## Grain 0.2.11 (July 2, 2025)

* New features:
  * Automatic publishing releases to PyPI via GitHub actions.
  * Nightly builds.
  * Introduced changelog.