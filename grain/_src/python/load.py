"""High level APIs that serve as a single endpoint for very common use cases."""

from typing import Optional

from grain._src.core import monitoring as grain_monitoring
from grain._src.core import sharding
from grain._src.core import transforms
from grain._src.core import usage_logging
from grain._src.python import data_loader
from grain._src.python import data_sources
from grain._src.python import options
from grain._src.python import samplers

from grain._src.core import monitoring


_api_usage_counter = monitoring.Counter(
    "/grain/python/load/api",
    monitoring.Metadata(description="API initialization counter."),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)


def load(
    source: data_sources.RandomAccessDataSource,
    *,
    num_epochs: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
    shard_options: sharding.ShardOptions = sharding.NoSharding(),
    transformations: transforms.Transformations = (),
    batch_size: Optional[int] = None,
    drop_remainder: bool = False,
    worker_count: Optional[int] = 0,
    read_options: Optional[options.ReadOptions] = None,
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
  _api_usage_counter.Increment("load")
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
        transforms.Batch(batch_size, drop_remainder=drop_remainder)
    )
  return data_loader.DataLoader(
      data_source=source,
      sampler=sampler,
      operations=transformations,
      worker_count=worker_count,
      read_options=read_options,
  )
