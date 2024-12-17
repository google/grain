# Data Sources



A Grain data source is responsible for retrieving individual records. Records
could be in a file/storage system or generated on the fly. Data sources need to
implement the following protocol:

```python
class RandomAccessDataSource(Protocol, Generic[T]):
  """Interface for datasources where storage supports efficient random access."""

  def __len__(self) -> int:
    """Number of records in the dataset."""

  def __getitem__(self, record_key: SupportsIndex) -> T:
    """Retrieves record for the given record_key."""
```

## File Format

Note that the underlying file format/storage system needs to support efficient
random access. Grain currently supports random-access file format [ArrayRecord](https://github.com/google/array_record).

## Available Data Sources

We provide a variety of data sources for Grain, which we discuss in the following sections.

### Range Data Source

This data source mimics the built-in Python
[range class](https://docs.python.org/3/library/functions.html#func-range). It
can be used for initial Grain testing or if your use case involves generating
records on the fly (for example if you only want to generate synthetic records
online rather than read records from storage.)

```python
range_data_source = grain.python.RangeDataSource(start=1,stop=10,step=2)
print(list(x)) # prints [1, 3, 5, 7, 9]
```

### ArrayRecord Data Source

This is a data source for [ArrayRecord](https://github.com/google/array_record) files.
The data source accepts a single/list of PathLike or File Instruction objects.

PathLike are objects implementing
[os.PathLike](https://docs.python.org/3/library/os.html#os.PathLike). For these
objects, the data source starts by opening the files to get the number of
records in each file. It uses this information to build a global index over all
files.

On the other hand, File Instruction objects are objects implementing the
following protocol:

```python
class FileInstruction(Protocol):
  """Protocol with same interface as FileInstruction objects returned by Tfds."""

  filename: str
  skip: int # Number of examples in the beginning of the shard to skip.
  take: int # Number of examples to include.
  examples_in_shard: int # Total number of records in the shard.
```

File instruction objects enable a few use cases:

*   Selecting only a subset of records within a file.
*   Saving startup time when the file sizes are known in advance (since the data
    source skips opening files in that case.)

### TFDS Data Source

TFDS provides Grain compatible data sources via `tfds.data_source()`.
Arguments are equivalent to `tfds.load()`. For more information see

```python
tfds_data_source = tfds.data_source("imagenet2012", split="train[:75%]")
```

## Implement your own Data Source

You can implement your own data source and use it with Grain. It needs to
implement the `RandomAccessDataSource` protocol defined above. In addition, you
need to pay attention to the following:

*   **Data Sources should be pickleable.** This is because in the multi-worker
    setting, data sources are pickled and sent to child processes, where each
    child process reads only the records it needs to process. File reader
    objects are usually not pickleable. In our data sources, we implement
    `__getstate__` and `__setstate__` to ensure that file readers aren't part of
    the state when the data source is pickled, but rather are recreated upon
    unpickling.
*   **Open file handles should be closed after use.** Data sources typically
    open underlying files in order to read records from them. We recommend
    implementing data sources as context managers that close their open file
    handles within the `__exit__` method. When opening a data source, the
    `DataLoader` will first attempt to use the data source as a context manager.
    If the data source doesn't implement the context manager protocol, it will
    be used as-is, without a `with` statement.
