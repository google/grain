"""GrainPool for LazyDataset, with batching in the parent process."""

import os
from typing import Any, Iterator

from absl import logging
import cloudpickle
from grain._src.python import options as grain_options
from grain._src.python.lazy_dataset import lazy_dataset


def _pickle_and_unpickle_lazy_dataset(
    lazy_ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
):
  """This prevents the user-specified LazyDataset from being mutated and also verifies that it is picklable."""
  try:
    pickled_lazy_ds = cloudpickle.dumps(lazy_ds)
    new_lazy_ds = cloudpickle.loads(pickled_lazy_ds)
    return new_lazy_ds
  except Exception as exc:
    raise ValueError("LazyDataset is not picklable.") from exc


class LazyDatasetDataLoader:
  """User-friendly DataLoader abstraction to parallelize processing of LazyDataset PyGrain pipelines among a set of processes."""

  def __init__(
      self,
      *,
      lazy_ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
      num_processes: int = 1,
      multithreading_read_options: grain_options.ReadOptions | None = None,
      elastic: bool = False,
  ):
    """Initialize a LazyDataset GrainPool.

    Args:
      lazy_ds: Function to apply to input elements.
      num_processes: Number of child processes.
      multithreading_read_options: Options to use for reading.
      elastic: Whether or not to use many-to-one in parent process only.
    """
    if num_processes is None:
      self._num_processes = os.cpu_count()
      if self._num_processes is None:
        raise NotImplementedError("Cannot determine the number of CPUs.")
    else:
      self._num_processes = num_processes
    logging.info("Grain pool will use %i processes.", self._num_processes)

    if multithreading_read_options is None:
      self._read_options = grain_options.ReadOptions(
          num_threads=20,
          prefetch_buffer_size=256,
      )
    else:
      self._read_options = multithreading_read_options

    # Don't mutate original lazydataset.
    self._lazy_ds = _pickle_and_unpickle_lazy_dataset(lazy_ds)
    self._elastic = elastic
    self._lazy_ds_iter = self._setup_multiprocessing_ds_iter()

  def _setup_multiprocessing_ds_iter(self) -> Iterator[Any]:
    """Create iterator for pipeline with desired concurrency settings."""
    multiprocessing_options = grain_options.MultiprocessingOptions(
        num_workers=self._num_processes, per_worker_buffer_size=100
    )
    if isinstance(self._lazy_ds, lazy_dataset.LazyMapDataset):
      # This already has elasticity.
      multiproc_iter_ds = lazy_dataset.MultiprocessPrefetchLazyIterDataset(
          self._lazy_ds.to_iter_dataset(), multiprocessing_options
      )
    else:  # LazyIterDataset
      if self._elastic:
        raise ValueError("Elasticity for LazyIterDatasets not supported yet.")
      else:
        multiproc_iter_ds = lazy_dataset.MultiprocessPrefetchLazyIterDataset(
            self._lazy_ds, multiprocessing_options
        )
    return iter(multiproc_iter_ds)

  def __iter__(self):
    """Return iterator for lazy_ds."""
    return self._lazy_ds_iter

  def __next__(self):
    try:
      return next(self._lazy_ds_iter)
    except StopIteration as exc:
      raise StopIteration() from exc

  def __del__(self):
    self._shutdown()

  def __enter__(self):
    return self._lazy_ds_iter

  def __exit__(self, exc_type, exc_value, exc_traceback):
    logging.info("Grain pool is exiting.")

  def _shutdown(self) -> None:
    logging.info("Shutting down GrainPool.")
    # TODO(laueric): Implement this.
