# Change log

The change log file hosting all releases with lists of new features and breaking
changes. Best viewed [here](https://google-grain.readthedocs.io/en/latest/changelog.html).

## Unreleased

* New features
  * Add `reseed_each_epoch` option to `MapDataset.repeat` that allows to replay
    the first epoch exactly if set to False (True by default).

* Breaking changes:

* Deprecations:
  * Graduate `grain.experimental.apply_transformations` to
   `grain.{MapDataset|IterDataset}.apply`. The experimental API will soon be
    deprecated.
* Bug fixes

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

* Breaking changes:
  * Upgrades `array_record` and `protobuf`.

* Deprecations:

* Bug fixes

## Grain 0.2.11 (July 2, 2025)

* New features:
  * Automatic publishing releases to PyPI via GitHub actions.
  * Nightly builds.
  * Introduced changelog.