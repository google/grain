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
"""TfGrain implementation of the LazyIterDataset API.

This is useful for migrating existing TfGrain pipelines to use PyGrain or for
modifying existing PyGrain pipelines to take advantage of TfGrain performance.

Usage example:

```
from grain._src.python.lazy_dataset.sources import tfgrain as tfgrain_lazy

data_loader = grain.TfDataLoader(source=source, sampler=sampler, ...)
dataset = tfgrain_lazy.TfGrainLazyIterDataset(data_loader)

iterator = iter(dataset)

saved_state = maybe_get_saved_state()
if saved_state:
  iterator.set_state(saved_state)

for batch in iterator:
  print(batch)
  save_state(iterator.get_state())
```
"""

import json
from typing import Any

from clu.data import dataset_iterator
from grain._src.python.lazy_dataset import lazy_dataset
import grain.tensorflow as tfgrain


class TfGrainLazyDatasetIterator(
    lazy_dataset.LazyDatasetIterator[dataset_iterator.Element]
):
  """LazyDatasetIterator backed by a TfGrain iterator."""

  def __init__(self, tfgrain_iterator: tfgrain.TfGrainDatasetIterator):
    self._tfgrain_iterator = tfgrain_iterator

  def __next__(self) -> dataset_iterator.Element:
    return next(self._tfgrain_iterator)

  def get_state(self) -> dict[str, Any]:
    raw_state = self._tfgrain_iterator.get_state()
    state = json.loads(raw_state.decode())
    if not isinstance(state, dict):
      raise ValueError(
          "Received unexpected state from tfgrain iterator. "
          "Expected a dict but got ",
          state,
      )
    for key in state:
      if not isinstance(key, str):
        raise ValueError(
            "Received unexpected state from tfgrain iterator. "
            "Expected a dict with string keys but got key ",
            key,
        )
    return state

  def set_state(self, state: dict[str, Any]):
    """Sets the current state of the iterator."""
    self._tfgrain_iterator.set_state(json.dumps(state).encode())


class TfGrainLazyIterDataset(
    lazy_dataset.LazyIterDataset[dataset_iterator.Element]
):
  """A PyGrain LazyIterDataset backed by a TfGrain DataLoader."""

  def __init__(self, dataset):
    self._dataset = dataset

  def __iter__(self) -> TfGrainLazyDatasetIterator:
    it = iter(self._dataset)
    if not isinstance(it, tfgrain.TfGrainDatasetIterator):
      raise ValueError(
          "Expected the dataset passed to TfGrainLazyIterDataset "
          "to produce iterators of type "
          "tfgrain.TfGrainDatasetIterator, but got an iterator of "
          "type ",
          type(it),
      )
    return TfGrainLazyDatasetIterator(it)
