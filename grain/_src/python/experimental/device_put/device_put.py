"""Utility to prefetch data on CPU and devices."""

from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch

ThreadPrefetchIterDataset = prefetch.ThreadPrefetchIterDataset
