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


def _get_max_workers(
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


@dataclasses.dataclass(slots=True)
class PerformanceConfig:
  multiprocessing_options: grain_options.MultiprocessingOptions | None = None


def pick_performance_config(
    ds: dataset.IterDataset,
    *,
    ram_budget_mb: int,
    max_workers: int | None,
    samples_to_check: int = 5,
) -> PerformanceConfig:
  """Analyzes element size to choose an optimal number of workers, then creates a MultiprocessPrefetchIterDataset."""
  num_workers = _get_max_workers(
      ds,
      ram_budget_mb=ram_budget_mb,
      max_workers=max_workers,
      samples_to_check=samples_to_check,
  )
  return PerformanceConfig(
      multiprocessing_options=grain_options.MultiprocessingOptions(
          num_workers=num_workers
      )
  )
