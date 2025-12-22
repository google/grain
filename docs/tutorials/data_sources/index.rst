.. _data-sources-tutorials-section:

Data Sources
============

A `Grain` data source is responsible for retrieving individual records. Records
could be in a file/storage system or generated on the fly. There are two main kinds of
data sources: those supporting efficient random access, and those only supporing a
sequential access and can be iteratated over. 

Data sources with random access need to implement the following protocol:

.. code-block:: python

    class RandomAccessDataSource(Protocol, Generic[T]):
      """Interface for datasources where storage supports efficient random access."""

      def __len__(self) -> int:
        """Number of records in the dataset."""

      def __getitem__(self, record_key: SupportsIndex) -> T:
        """Retrieves record for the given record_key."""

Data sources / datasets with no random access should implement the ``grain.IterDataset``
(see the *Dataset basics* page for further details)

.. code-block:: python

    class IterDataset(_Dataset, Iterable[T]):
      """Interface for datasets which can be iterated over."""
      def __iter__(self) -> DatasetIterator[T]:
        """Returns an iterator for this dataset."""



File formats and available Data Sources
---------------------------------------

The underlying file format/storage system needs to support efficient random access.
We provide a variety of data sources for `Grain`, which we discuss in the
:ref:`tutorials-label` section below.


Range Data Source
-----------------

This data source mimics the built-in Python
`range class <https://docs.python.org/3/library/functions.html#func-range>`_. It
can be used for initial `Grain` testing or if your use case involves generating
records on the fly (for example if you only want to generate synthetic records
online rather than read records from storage.)

.. code-block:: python

    range_data_source = grain.python.RangeDataSource(start=1,stop=10,step=2)
    print(list(range_data_source)) # prints [1, 3, 5, 7, 9]


Implement your own Data Source
------------------------------

You can implement your own data source and use it with `Grain`. It needs to
implement one of the ``RandomAccessDataSource`` or ``IterDataset`` protocols
defined above. In addition, you need to pay attention to the following:

*   **Data Sources should be pickleable.** This is because in the multi-worker
    setting, data sources are pickled and sent to child processes, where each
    child process reads only the records it needs to process. File reader
    objects are usually not pickleable. In our data sources, we implement
    ``__getstate__`` and ``__setstate__`` to ensure that file readers aren't part of
    the state when the data source is pickled, but rather are recreated upon
    unpickling.
*   **Open file handles should be closed after use.** Data sources typically
    open underlying files in order to read records from them. We recommend
    implementing data sources as context managers that close their open file
    handles within the ``__exit__`` method. When opening a data source, the
    ``DataLoader`` will first attempt to use the data source as a context manager.
    If the data source doesn't implement the context manager protocol, it will
    be used as-is, without a ``with`` statement.



.. _tutorials-label:

Tutorials
=========

This section contains tutorials for using Grain to read data from various sources.

.. toctree::
   :maxdepth: 1

   parquet_dataset_tutorial.md
   arrayrecord_data_source_tutorial.md
   bagz_data_source_tutorial.md
   huggingface_dataset_tutorial.md
   pytorch_dataset_tutorial.md

File systems
------------

.. toctree::
   :maxdepth: 1

   load_from_s3_tutorial.md
   load_from_gcs_tutorial.md


