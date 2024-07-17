# Copyright 2024 Google LLC
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
"""DataLoader for LazyDataset."""
from typing import Optional, Union

import cloudpickle
from grain._src.python import options as grain_options
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import prefetch


def _confirm_picklable_and_copy_lazy_dataset(
    lazy_ds: Union[lazy_dataset.MapDataset, lazy_dataset.IterDataset],
):
  """Makes a copy of the given LazyDataset through pickling roundtrip."""
  try:
    pickled_lazy_ds = cloudpickle.dumps(lazy_ds)
    return cloudpickle.loads(pickled_lazy_ds)
  except Exception as e:
    raise ValueError(f"{lazy_ds} is not picklable.") from e


class DataLoader:
  """Reads and transforms data as described by the given LazyDataset."""

  def __init__(
      self,
      *,
      lazy_ds: Union[lazy_dataset.MapDataset, lazy_dataset.IterDataset],
      multiprocessing_options: Optional[
          grain_options.MultiprocessingOptions
      ] = None,
      read_options: Optional[grain_options.ReadOptions] = None,
  ):
    """Initializes DataLoader.

    Args:
      lazy_ds: User-defined MapDataset or IterDataset.
      multiprocessing_options: Options to use for executing LazyDataset pipeline
        in multiple processes.
      read_options: Options to use for reading data from disk.
    """
    self._multiprocessing_options = multiprocessing_options

    # Avoid mutating the original LazyDataset.
    lazy_ds = _confirm_picklable_and_copy_lazy_dataset(lazy_ds)
    self._iter_ds = (
        lazy_ds.to_iter_dataset(read_options)
        if isinstance(lazy_ds, lazy_dataset.MapDataset)
        else lazy_ds
    )

    if self._multiprocessing_options:
      self._iter_ds = prefetch.MultiprocessPrefetchIterDataset(
          self._iter_ds, self._multiprocessing_options
      )

  def __iter__(self):
    """Return iterator for lazy_ds."""
    return iter(self._iter_ds)
