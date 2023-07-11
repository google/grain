"""High level APIs that serve as a single endpoint for very common use cases."""

from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import data_loader
from grain._src.python import data_sources
from grain._src.python import operations
from grain._src.python import options
from grain._src.python import samplers


def load(
    source: data_sources.RandomAccessDataSource,
    *,
    num_epochs: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shard_options: sharding.ShardOptions = sharding.NoSharding(),
    transformations: transforms.Transformations = (),
    batch_size: int | None = None,
    drop_remainder: bool = False,
    worker_count: int | None = 0,
    read_options: options.ReadOptions | None = None,
) -> data_loader.DataLoader:
  """Convenient method for simple pipelines on top of a data source.

  Args:
    source: Data source to load from. This can be one of the file data sources
      provided by Grain, a TFDS data source (`tfds.data_source(...)`) or your
      custom data source.
    num_epochs: See IndexSampler.
    shuffle: See IndexSampler.
    seed: See IndexSampler.
    shard_options: See IndexSampler.
    transformations: List of local (stateless) transformations:
    batch_size: Optional batch size. If provided will apply BatchOperation().
    drop_remainder: Whether to drop partial batches.
    worker_count: Number of child processes launched to parallelize the
      transformations among. Zero means processing runs in the same process.
    read_options: Read options for the data loader. See ReadOptions.

  Returns:
    DataLoader for this dataset.
  """
  usage_logging.log_event("load", tag_3="PyGrain")
  sampler = samplers.IndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=shard_options,
  )
  if batch_size is not None:
    transformations = list(transformations)
    transformations.append(
        operations.BatchOperation(batch_size, drop_remainder=drop_remainder)
    )
  return data_loader.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
      read_options=read_options,
  )
