# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to prefetch data on CPU and devices."""

from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch

ThreadPrefetchIterDataset = prefetch.ThreadPrefetchIterDataset


def device_put(
    ds: dataset.IterDataset,
    device,
    *,
    cpu_buffer_size: int = 4,
    device_buffer_size: int = 2,
) -> dataset.IterDataset:
  """Moves the data to the given devices with prefetching.

  Stage 1: A CPU-side prefetch buffer.
  Stage 2: Per-device buffers for elements already transferred to the device.

  Args:
    ds: Dataset to prefetch.
    device: same arguments as in jax.device_put.
    cpu_buffer_size: Number of elements to prefetch on CPU.
    device_buffer_size: Number of elements to prefetch per device.

  Returns:
    Dataset with the elements prefetched to the devices.
  """
  ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cpu_buffer_size)
  # May raise ImportError if jax is not linked.
  import jax  # pylint:disable=g-import-not-at-top  # pytype:disable=import-error

  ds = ds.map(lambda x: jax.device_put(x, device))
  ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=device_buffer_size)
  return ds
