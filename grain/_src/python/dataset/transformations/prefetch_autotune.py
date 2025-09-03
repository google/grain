"""Autotuning for the multiprocess prefetcher."""

import dataclasses
import itertools
import math
import multiprocessing
import os
import sys
from absl import logging
from grain._src.python import options as grain_options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch
import numpy as np

cpu_count = multiprocessing.cpu_count


def _get_element_size_bytes(element):
  """Recursively calculate the deep total size of the object."""
  size = 0
  if isinstance(element, dict):
    for _, value in element.items():
      size += _get_element_size_bytes(value)
  elif isinstance(element, (list, tuple)):
    for item in element:
      size += _get_element_size_bytes(item)
  elif isinstance(element, np.ndarray):
    size += element.nbytes
  else:
    # For other Python objects, sys.getsizeof is a reasonable approximation.
    size += sys.getsizeof(element)
  return size


def _get_average_element_size_mb(
    dss: list[dataset.IterDataset],
    samples_to_check: int = 5,
):
  """Calculates the average size of elements in the given datasets.

  Args:
    dss: A list of datasets to sample from.
    samples_to_check: The number of samples to check to estimate element size.

  Returns:
    The average size of elements in megabytes.
  """
  if not dss:
    return 0
  samples_per_ds = math.ceil(float(samples_to_check) / len(dss))
  # Sample elements to measure their size.
  elements_to_sample = []
  elements_to_sample.extend(
      itertools.chain.from_iterable(
          map(lambda ds: itertools.islice(ds, samples_per_ds), dss)
      )
  )
  if len(elements_to_sample) < samples_to_check:
    logging.warning('Warning: Not enough elements to sample.')
    return 0

  avg_size_bytes = np.mean(
      [_get_element_size_bytes(e) for e in elements_to_sample]
  )
  avg_size_mb = float(avg_size_bytes) / (1024 * 1024)
  return avg_size_mb


def _get_num_workers(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_workers: int | None,
    samples_to_check: int = 5,
):
  """Analyzes element size to choose an optimal number of workers, then creates a MultiprocessPrefetchIterDataset."""
  average_elem_size_mb = _get_average_element_size_mb(
      [ds], samples_to_check=samples_to_check
  )
  if max_workers is None:
    # MultiprocessPrefetchIterDataset defaults to using all CPUs.
    max_workers = cpu_count()

  if average_elem_size_mb <= 0:
    logging.warning(
        'Warning: Average element size is zero. Defaulting to max workers.'
    )
    return max_workers
  num_workers = int(ram_budget_mb / average_elem_size_mb)

  return min(num_workers, max_workers)


def _get_buffer_size(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_buffer_size: int | None,
    samples_to_check: int = 5,
):
  """Analyzes elements' sizes to choose an optimal buffer size for a PrefetchIterDataset."""
  # Find the datasets with non-batched elements.
  dss = _find_prefetch_iter_dataset_parents(ds)
  average_elem_size_mb = _get_average_element_size_mb(
      dss, samples_to_check=samples_to_check
  )
  if max_buffer_size is None:
    max_buffer_size = 1000

  if average_elem_size_mb <= 0:
    logging.warning(
        'Warning: Average element size is zero. Defaulting to max threads.'
    )
    return grain_options.ReadOptions(prefetch_buffer_size=max_buffer_size)

  # Ram budget needs to be devided by the number of `PrefetchIterDataset`s,
  # as all of them will be prefetched in parallel. We simplify and use the
  # average, ideally each `PrefetchIterDataset` should have its own buffer size.
  buffer_size = int(ram_budget_mb / (average_elem_size_mb * len(dss)))

  return grain_options.ReadOptions(
      prefetch_buffer_size=min(buffer_size, max_buffer_size),
  )


@dataclasses.dataclass(slots=True)
class PerformanceConfig:
  multiprocessing_options: grain_options.MultiprocessingOptions | None = None
  read_options: grain_options.ReadOptions | None = None


def pick_performance_config(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int | None,
    max_workers: int | None,
    max_buffer_size: int | None,
    samples_to_check: int = 5,
) -> PerformanceConfig:
  """Analyzes element size to choose an optimal number of workers for a MultiprocessPrefetchIterDataset.

  Args:
    ds: The input dataset.
    ram_budget_mb: The user predicted RAM budget in megabytes.
    max_workers: The maximum number of processes to use.
    max_buffer_size: The maximum buffer size to use.
    samples_to_check: The number of samples to check to estimate element size.

  Returns:
    A PerformanceConfig object containing the optimal number of workers.
  """

  num_workers = _get_num_workers(
      ds,
      ram_budget_mb=ram_budget_mb,
      max_workers=max_workers,
      samples_to_check=samples_to_check,
  )
  read_options = _get_buffer_size(
      ds,
      ram_budget_mb=ram_budget_mb,
      max_buffer_size=max_buffer_size,
      samples_to_check=samples_to_check,
  )
  return PerformanceConfig(
      multiprocessing_options=grain_options.MultiprocessingOptions(
          num_workers=num_workers
      ),
      read_options=read_options,
  )


def _find_prefetch_iter_dataset_parents(
    ds: dataset.IterDataset | dataset.MapDataset,
) -> list[dataset.IterDataset | dataset.MapDataset | None]:
  """Finds the parents of PrefetchIterDatasets.

  Used for finding dataset with non-batched elements.

  Args:
    ds: The input dataset.

  Returns:
    The list of parents of PrefetchIterDataset or None if no PrefetchIterDataset
    is found.
  """
  prefetch_iter_dataset_parents = []
  if isinstance(ds, prefetch.PrefetchIterDataset):
    prefetch_iter_dataset_parents.append(ds.parents[0])
    return prefetch_iter_dataset_parents
  for parent in ds.parents:
    prefetch_iter_dataset_parents.extend(
        _find_prefetch_iter_dataset_parents(parent)
    )
  return prefetch_iter_dataset_parents
