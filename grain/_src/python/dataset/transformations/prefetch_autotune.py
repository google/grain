"""Autotuning for the multiprocess prefetcher."""

import dataclasses
import itertools
import multiprocessing
import sys

from absl import logging
from grain._src.python import options as grain_options
from grain._src.python.dataset import dataset
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


def _get_num_workers(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_workers: int | None,
    samples_to_check: int = 5,
):
  """Analyzes element size to choose an optimal number of workers, then creates a MultiprocessPrefetchIterDataset."""
  if max_workers is None:
    # MultiprocessPrefetchIterDataset defaults to using all CPUs.
    max_workers = cpu_count()

  # Sample elements to measure their size.
  elements_to_sample = list(itertools.islice(ds, samples_to_check))
  if len(elements_to_sample) < samples_to_check:
    logging.warning(
        "Warning: Not enough elements to sample. Defaulting to max workers."
    )
    return max_workers

  avg_size_bytes = np.mean(
      [_get_element_size_bytes(e) for e in elements_to_sample]
  )
  avg_size_mb = float(avg_size_bytes) / (1024 * 1024)
  if avg_size_mb <= 0:
    logging.warning(
        "Warning: Average element size is zero. Defaulting to max workers."
    )
    return max_workers
  num_workers = int(ram_budget_mb / avg_size_mb)

  return min(num_workers, max_workers)


def _get_num_threads(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_threads: int | None,
    prefetch_buffer_size: int = 500,
    samples_to_check: int = 5,
):
  """Analyzes element size to choose an optimal number of threads for a PrefetchIterDataset."""

  if max_threads is None:
    max_threads = 16

  # Sample elements to measure their size.
  elements_to_sample = list(
      itertools.islice(ds, samples_to_check + prefetch_buffer_size)
  )
  if len(elements_to_sample) < samples_to_check + prefetch_buffer_size:
    logging.warning(
        "Warning: Not enough elements to sample. Defaulting to max number of"
        " threads."
    )
    return grain_options.ReadOptions(
        num_threads=max_threads, prefetch_buffer_size=prefetch_buffer_size
    )

  elements_sizes = [_get_element_size_bytes(e) for e in elements_to_sample]

  buffer_size_bytes = [
      sum(elements_sizes[i : i + prefetch_buffer_size])
      for i in range(samples_to_check)
  ]

  avg_buffer_bytes = np.mean(buffer_size_bytes)
  avg_buffer_mb = float(avg_buffer_bytes) / (1024 * 1024)
  if avg_buffer_mb <= 0:
    logging.warning(
        "Warning: Average element size is zero. Defaulting to max threads."
    )
    return grain_options.ReadOptions(
        num_threads=max_threads, prefetch_buffer_size=prefetch_buffer_size
    )
  num_threads = int(ram_budget_mb / avg_buffer_mb)

  return grain_options.ReadOptions(
      num_threads=min(num_threads, max_threads),
      prefetch_buffer_size=prefetch_buffer_size,
  )


@dataclasses.dataclass(slots=True)
class PerformanceConfig:
  multiprocessing_options: grain_options.MultiprocessingOptions | None = None
  read_options: grain_options.ReadOptions | None = None


def pick_performance_config(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_workers: int | None,
    max_threads: int | None,
    samples_to_check: int = 5,
) -> PerformanceConfig:
  """Analyzes element size to choose an optimal number of workers for a MultiprocessPrefetchIterDataset.

  and number of threads for a PrefetchIterDataset.

  Args:
    ds: The input dataset.
    ram_budget_mb: The RAM budget in megabytes.
    max_workers: The maximum number of workers to use.
    max_threads: The maximum number of threads to use.
    samples_to_check: The number of samples to check to estimate element size.

  Returns:
    A PerformanceConfig object containing the optimal number of workers and
    threads.
  """
  # Variable name `workers` refers to processes and is used for historical
  # reasons.
  num_workers = _get_num_workers(
      ds,
      ram_budget_mb=ram_budget_mb,
      max_workers=max_workers,
      samples_to_check=samples_to_check,
  )
  read_options = _get_num_threads(
      ds,
      ram_budget_mb=ram_budget_mb,
      max_threads=max_threads,
      samples_to_check=samples_to_check,
  )
  return PerformanceConfig(
      multiprocessing_options=grain_options.MultiprocessingOptions(
          num_workers=num_workers
      ),
      read_options=read_options,
  )
