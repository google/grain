
Protocol
================

Grain data sources are responsible for retrieving individual records. Records
could be in a storage system or generated on the fly. There are two main kinds
of data sources: those supporting efficient random access (RA), and those only
supporing sequential access (SA).

One notable feature enabled by random access is global index shuffle without
loading the dataset into memory. Datasets that only support sequential access
must be shuffled in windows and loaded one window of values at a time.

Sources with RA
---------------

Data sources with random access need to implement the following protocol:

.. code-block:: python

  class RandomAccessDataSource(Protocol[T]):
    """Interface for datasets where storage supports efficient random access."""

    def __len__(self) -> int:
      """Returns the total number of records in the data source."""

    def __getitem__(self, index: int) -> T:
      """Returns the value for the given index.

      This method must be thread-safe and deterministic.

      Arguments:
        index: An integer in `[0, len(self)-1]`.

      Returns:
        The corresponding record. File data sources often return the raw bytes
        but records can be any Python object.
      """

Supported RA formats
^^^^^^^^^^^^^^^^^^^^

The simplest in-memory format supporting the ``RandomAccessDataSource`` protocol
is ``collections.abc.Sequence`` -- it is useful for testing since you can pass a
list or a tuple directly. We also offer ``grain.sources.RangeDataSource`` as
shortcut source that mimics the behavior of ``range()``.

Following are the supported randomly accessible on-disk formats:

*  :doc:`Bagz <../tutorials/data_sources/bagz_data_source_tutorial>`
*  :doc:`ArrayRecord <../tutorials/data_sources/arrayrecord_data_source_tutorial>`
*  :doc:`HuggingFace dataset <../tutorials/data_sources/huggingface_dataset_tutorial>`
*  :doc:`PyTorch dataset <../tutorials/data_sources/pytorch_dataset_tutorial>`

Sources with SA
---------------

Data sources with no random access should implement a combination of
``grain.IterDataset`` and ``grain.DatasetIterator``
(see
`Dataset basics <https://google-grain.readthedocs.io/en/latest/tutorials/dataset_basic_tutorial.html#iterdataset>`__
for more details).

.. code-block:: python

    class IterDataset(Iterable[T]):
      """Interface for datasets which can be iterated over."""

      def __iter__(self) -> DatasetIterator[T]:
        """Returns an iterator for this dataset."""

    class DatasetIterator(Iterator[T], abc.ABC):
      """``IterDataset`` iterator."""

      def get_state(self) -> dict[str, Any]:
        """Returns the current state of the iterator for checkpointing."""

      def set_state(self, state: dict[str, Any]) -> None:
        """Recovers the iterator to the given state from a checkpoint."""

Sequential sources can only be used with ``Dataset`` API.

Supported SA formats
^^^^^^^^^^^^^^^^^^^^

We provide sources for the following sequential data formats:

*  :doc:`Parquet <../tutorials/data_sources/parquet_dataset_tutorial>`
*  `TfRecord <https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.TFRecordIterDataset.html#grain.experimental.TFRecordIterDataset>`__


Implement your own source
------------------------------

You can implement your own data source and use it with Grain. It needs to
implement one of the ``RandomAccessDataSource`` or ``IterDataset`` protocols
defined above. In addition, you need to pay attention to the following:

*   Data sources should be pickleable. This is because in the multi-worker
    setting, data sources are pickled and sent to child processes, where each
    child process reads only the records it needs to process. File reader
    objects are usually not pickleable. In our data sources, we implement
    ``__getstate__`` and ``__setstate__`` to ensure that file readers aren't
    part of the state when the data source is pickled, but rather are recreated
    upon unpickling.
*   If used with ``DataLoader`` sources should implement ``__repr__``. This is
    needed for ``DataLoader`` checkpoint validation.


File systems
------------

Grain supports the formats mentioned above in combination with the following
file systems (in addition to the local file system):

*  :doc:`S3 <../tutorials/data_sources/load_from_s3_tutorial>`
*  :doc:`GCS <../tutorials/data_sources/load_from_gcs_tutorial>`
