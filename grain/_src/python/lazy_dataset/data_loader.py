"""DataLoader for LazyDataset."""

import cloudpickle
from grain._src.python import options as grain_options
from grain._src.python.lazy_dataset import lazy_dataset


def _confirm_picklable_and_copy_lazy_dataset(
    lazy_ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
):
  """This function may be used to prevent the user-specified LazyDataset from being mutated."""
  try:
    pickled_lazy_ds = cloudpickle.dumps(lazy_ds)
    new_lazy_ds = cloudpickle.loads(pickled_lazy_ds)
    return new_lazy_ds
  except Exception as e:
    raise ValueError("LazyDataset is not picklable.") from e


class DataLoader:
  """User-friendly DataLoader abstraction to parallelize processing of LazyDataset PyGrain pipelines among a set of processes."""

  def __init__(
      self,
      *,
      lazy_ds: lazy_dataset.LazyMapDataset | lazy_dataset.LazyIterDataset,
      multiprocessing_options: (
          grain_options.MultiprocessingOptions | None
      ) = None,
      read_options: grain_options.ReadOptions | None = None,
  ):
    """Initialize a LazyDataset GrainPool.

    Args:
      lazy_ds: User-defined LazyMapDataset or LazyIterDataset.
      multiprocessing_options: Options to use for executing LazyDataset pipeline
        on multiprocessing.
      read_options: Options to use for reading data from disk.
    """
    self._multiprocessing_options = multiprocessing_options

    if read_options:
      self._read_options = read_options
    else:
      # num_threads = 16, prefetch_buffer_size=500
      self._read_options = grain_options.ReadOptions()

    # Don't mutate original lazydataset.
    self._lazy_ds = _confirm_picklable_and_copy_lazy_dataset(lazy_ds)

    if self._multiprocessing_options:
      self._iter_ds = self._set_up_multiprocessing_iter_ds()
    else:
      self._iter_ds = (
          self._lazy_ds.to_iter_dataset()
          if isinstance(self._lazy_ds, lazy_dataset.LazyMapDataset)
          else self._lazy_ds
      )

  def _set_up_multiprocessing_iter_ds(self) -> lazy_dataset.LazyIterDataset:
    """Create iterator for pipeline with desired concurrency settings."""
    if isinstance(self._lazy_ds, lazy_dataset.LazyMapDataset):
      # This already has elasticity.
      return lazy_dataset.MultiprocessPrefetchLazyIterDataset(
          self._lazy_ds.to_iter_dataset(), self._multiprocessing_options
      )

    # LazyIterDataset without elasticity
    return lazy_dataset.MultiprocessPrefetchLazyIterDataset(
        self._lazy_ds, self._multiprocessing_options
    )

  def __iter__(self):
    """Return iterator for lazy_ds."""
    return iter(self._iter_ds)
