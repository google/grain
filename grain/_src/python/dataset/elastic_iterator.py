# Copyright 2025 Google LLC
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
"""Iterator supporting changes in the number of hosts (dataset shards)."""

import copy
import functools
from typing import Any, TypeVar

from grain._src.core import sharding
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import (
    filter as filter_dataset,
)
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import mix
from grain._src.python.dataset.transformations import prefetch
from grain._src.python.dataset.transformations import (
    zip as zip_dataset,
)

T = TypeVar("T")

_GLOBAL_NEXT_INDEX_STATE_KEY = "global_next_index"


def _verify_transformations_supported(ds: dataset.IterDataset) -> None:
  """Checks for unsupported transformations in the dataset graph.

  Args:
    ds: The dataset to check.

  Raises:
    ValueError: If an unsupported transformation is found in the dataset graph.
  """
  to_check = [ds]
  while to_check:
    next_ds = to_check.pop()
    if isinstance(
        next_ds,
        (
            zip_dataset.ZipIterDataset,
            prefetch.PrefetchIterDataset,
            mix.MixedIterDataset,
        ),
    ):
      raise ValueError(
          "ElasticIterator for IterDataset does not support zip, mix or"
          " prefetch transformation yet."
      )
    to_check.extend(next_ds.parents)


def _find_sliceable_dataset(
    ds: dataset.IterDataset,
) -> dataset.IterDataset | None:
  """Finds the first sliceable dataset in the dataset graph.

  Args:
    ds: The dataset to search.

  Returns:
    The first sliceable dataset found, or None if no such dataset is found.
  """
  if isinstance(ds, prefetch.SupportsInPlaceSlicing):
    return ds
  if not hasattr(ds, "parents"):
    return None
  for parent in ds.parents:
    if not isinstance(parent, dataset.IterDataset):
      continue
    if found := _find_sliceable_dataset(parent):
      return found
  return None


def _find_sliceable_iterator(
    it: dataset.DatasetIterator,
) -> dataset.DatasetIterator | None:
  """Finds the first sliceable iterator in the iterator graph.

  This function recursively searches through the parents of the given iterator
  to find an iterator that supports `prefetch.SupportsSlicedStateManagement`.

  Args:
    it: The starting DatasetIterator.

  Returns:
    The first sliceable DatasetIterator found, or None if no such iterator
    is found in the graph.
  """
  if isinstance(it, prefetch.SupportsSlicedStateManagement):
    return it
  if not hasattr(it, "_parents"):
    return None
  for parent in it._parents:  # pylint: disable=protected-access
    if found := _find_sliceable_iterator(parent):
      return found
  return None


class ElasticIterDatasetIterator(dataset.DatasetIterator):
  """Iterator for IterDatasets in ElasticIterator.

  Helper class that adds elasticity to IterDatasets containing an underlying
  InterleaveIterDataset in their graph. It batches the dataset based on the
  global batch size and shard count, and coordinates distributed checkpointing
  and recovery across hosts using sliceable dataset state management.

  Attributes:
    shard_options: Sharding configuration specifying shard index and count.
    global_batch_size: The global batch size. Each shard will produce
      global_batch_size / shard_count elements in each step.
    read_options: The read options for prefetching.
  """

  def __init__(
      self,
      parent: dataset.IterDataset,
      global_batch_size: int,
      shard_options: sharding.ShardOptions,
      *,
      read_options: options.ReadOptions,
  ):
    """Initializes the ElasticIterDatasetIterator.

    Args:
      parent: The IterDataset to make elastic.
      global_batch_size: The global batch size.
      shard_options: The sharding configuration.
      read_options: The read options.

    Raises:
      ValueError: If no sliceable dataset is found in the graph, or if the
        global batch size is not divisible by the shard count, or if no
        sliceable iterator is found.
    """
    super().__init__()
    self._ds = parent
    if _find_sliceable_dataset(parent) is None:
      raise ValueError(
          "ElasticIterDatasetIterator requires an InterleaveIterDataset in the"
          " dataset graph."
      )

    self._global_batch_size = global_batch_size
    self._shard_options = shard_options
    self._read_options = read_options

    # These will be initialized when the iterator is created.
    self._closed = False
    ds = self._ds
    if self._global_batch_size > 0:
      host_batch_size, remainder = divmod(
          self._global_batch_size, self._shard_options.shard_count
      )
      if remainder:
        raise ValueError(
            f"Global batch size {self._global_batch_size} is not divisible by"
            f" shard count {self._shard_options.shard_count}."
        )
      ds = ds.batch(host_batch_size, drop_remainder=False)

    self._iterator = ds.__iter__()
    self._sliceable_iterator = _find_sliceable_iterator(self._iterator)
    if self._sliceable_iterator is None:
      raise ValueError(
          "ElasticIterDatasetIterator requires a sliceable iterator in the"
          " dataset graph."
      )

  @property
  def shard_options(self) -> sharding.ShardOptions:
    """The sharding configuration."""
    return self._shard_options

  def close(self) -> None:
    """Closes the iterator and releases any resources."""
    if self._closed:
      return
    self._closed = True
    self._iterator.close()

  def __next__(self) -> Any:
    """Returns the next element from the iterator."""
    return next(self._iterator)

  def get_shard_states(self) -> dict[int, Any]:
    """Returns the sharded states of the underlying sliceable iterator.

    The keys of the returned dictionary are global shard indices, and the values
    are the states for those specific global shards. The shard state comes from
    the highest sliceable iterator in the iterator graph.

    Returns:
      A dictionary mapping global shard indices to their states.
    """
    if not isinstance(
        self._sliceable_iterator, prefetch.SupportsSlicedStateManagement
    ):
      raise ValueError(
          "ElasticIterDatasetIterator does not support"
          " prefetch.SupportsSlicedStateManagement."
      )
    iter_shard_states = self._sliceable_iterator.get_shard_states()
    state_by_shard_index = {}
    for local_index, state in enumerate(iter_shard_states):
      global_shard_index = (
          local_index * self._shard_options.shard_count
          + self._shard_options.shard_index
      )
      state_by_shard_index[global_shard_index] = state
    return state_by_shard_index

  def set_shard_states(self, state: dict[int, Any]) -> None:
    """Sets the sharded states for the underlying sliceable iterator.

    Args:
      state: The state to set.

    This method assumes that the right shards are passed through to this
    iterator based on the shard options with `global_shard_states`.
    The input `state` is expected to be a dictionary containing:
      - "ds_iterator_states": A dict mapping global shard indices to their
        respective states.
      - "shard_count": The total number of shards.
    """
    ds = self._ds
    if self._global_batch_size > 0:
      host_batch_size, _ = divmod(
          self._global_batch_size, self._shard_options.shard_count
      )
      ds = ds.batch(host_batch_size, drop_remainder=False)
    if "_iterator" in self.__dict__:
      self._iterator.close()
    self._iterator = ds.__iter__()
    self._sliceable_iterator = _find_sliceable_iterator(self._iterator)
    self._closed = False

    if not isinstance(
        self._sliceable_iterator, prefetch.SupportsSlicedStateManagement
    ):
      raise ValueError(
          "ElasticIterDatasetIterator only supports"
          " prefetch.SupportsSlicedStateManagement."
      )
    ds_iterator_states = {int(k): v for k, v in state.items()}
    # We need to sort the states by the global shard index to ensure that the
    # states are set in the correct order.
    host_states = [state for _, state in sorted(ds_iterator_states.items())]
    self._sliceable_iterator.set_shard_states(host_states)

  def get_state(self) -> Any:
    """Returns the state of the inner iterator.

    This is only intended to be used when the ElasticIterator changes its
    topology or the number of hosts.
    """
    return self._iterator.get_state()

  def set_state(self, state: Any) -> None:
    """Sets the state of the inner iterator.

    This is only intended to be used when the ElasticIterator changes its
    topology or the number of hosts.

    Args:
      state: The state to set.
    """
    self._iterator.set_state(state)


class _ElasticMapDatasetIterator(dataset.DatasetIterator):
  """Iterator for MapDatasets in ElasticIterator.

  Internal helper class that adds elasticity to MapDatasets, slicing the dataset
  based on the shard index and count. It also batches the dataset based on the
  global batch size and shard count.

  Attributes:
    ds: The MapDataset to add elasticity to.
    global_batch_size: The global batch size.
    shard_options: The shard options.
    global_next_index: The next global index to use for slicing the dataset.
    read_options: The read options.
    multiprocessing_options: The multiprocessing options.
  """

  def __init__(
      self,
      ds: dataset.MapDataset,
      global_batch_size: int,
      shard_options: sharding.ShardOptions,
      *,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    super().__init__()
    self._ds = ds
    self._global_batch_size = global_batch_size
    self._shard_options = shard_options
    self._global_next_index = 0
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options

  @functools.cached_property
  def _iterator(self) -> dataset.DatasetIterator:
    ds = self._ds[
        self._global_next_index
        + self._shard_options.shard_index :: self._shard_options.shard_count
    ]
    host_batch_size, remainder = divmod(
        self._global_batch_size, self._shard_options.shard_count
    )
    if remainder:
      raise ValueError(
          f"Global batch size {self._global_batch_size} is not divisible by"
          f" shard count {self._shard_options.shard_count}."
      )
    ds = ds.batch(host_batch_size, drop_remainder=True)
    ds = ds.to_iter_dataset(read_options=self._read_options)
    if self._multiprocessing_options is not None:
      ds = ds.mp_prefetch(self._multiprocessing_options)
    return ds.__iter__()

  def __next__(self) -> Any:
    result = next(self._iterator)
    self._global_next_index += self._global_batch_size
    return result

  def get_state(self) -> dict[str, Any]:
    return {
        _GLOBAL_NEXT_INDEX_STATE_KEY: self._global_next_index,
    }

  def set_state(self, state):
    self._global_next_index = state[_GLOBAL_NEXT_INDEX_STATE_KEY]
    if "_iterator" in self.__dict__:
      self.__dict__["_iterator"].close()
    self.__dict__.pop("_iterator", None)

  def close(self) -> None:
    if "_iterator" in self.__dict__:
      self._iterator.close()


class ElasticIterator(dataset.DatasetIterator):
  """Iterator supporting recovery from a checkpoint after changes in sharding.

  The input dataset is expected to be unbatched and unsharded. In order to
  provide elasticity guarantee this iterator includes both, batching and
  sharding. This iterator explicitly disallows many-to-one transformations
  without a fixed ratio, like `filter` and generic `IterDataset`
  transformations. The implementation differs for MapDatasets and IterDatasets.

  MapDatasets:

  The iterator supports elastic re-configuration by having each
  shard produce the same exact checkpoint (while producing different data) as
  long as they are advanced the same number of steps.

  State of any shard can be used to restore the state of all of the shards after
  changes in sharding and global batch size.

  IterDatasets:

  IterDatasets support is still under development and comes with a few
  limitations. This class does not guarantee determinism between scaling. The
  limit of parallelism is the number of shards. The current implementation
  doesn't support multiprocessing.
  """

  def __init__(
      self,
      ds: dataset.MapDataset | dataset.IterDataset,
      global_batch_size: int,
      shard_options: sharding.ShardOptions,
      *,
      read_options: options.ReadOptions = options.ReadOptions(),
      multiprocessing_options: options.MultiprocessingOptions | None = None,
  ):
    """Initializes the ElasticIterator.

    Args:
      ds: The dataset to make elastic.
      global_batch_size: The global batch size.
      shard_options: The shard options.
      read_options: The read options.
      multiprocessing_options: The multiprocessing options.
    """
    super().__init__()
    to_check = [ds]
    while to_check:
      next_ds = to_check.pop()
      if isinstance(next_ds, filter_dataset.FilterMapDataset):
        raise ValueError(
            "ElasticIterator does not support `filter` transformation."
        )
      to_check.extend(next_ds.parents)

    self._shard_options = shard_options
    self._global_batch_size = global_batch_size
    self._read_options = read_options
    self._multiprocessing_options = multiprocessing_options
    if isinstance(ds, dataset.IterDataset):
      if multiprocessing_options:
        raise NotImplementedError(
            "ElasticIterator does not support multiprocess prefetching for"
            " IterDataset."
        )
      _verify_transformations_supported(ds)
      # We must set the slice on the original dataset so that the interleave
      # iterator is created with the correct (sliced) datasets.
      self._ds = copy.deepcopy(ds)
      prefetch._set_slice_iter_dataset(
          self._ds,
          slice(shard_options.shard_index, None, shard_options.shard_count),
      )
      if _find_sliceable_dataset(self._ds) is None:
        if self._read_options.num_threads > 1:
          datasets = []
          for thread in range(self._read_options.num_threads):
            d = copy.deepcopy(self._ds)
            prefetch._set_slice_iter_dataset(
                d, slice(thread, None, self._read_options.num_threads)
            )
            datasets.append(d)
          self._ds = interleave.InterleaveIterDataset(
              datasets, cycle_length=len(datasets)
          )
    else:
      self._ds = ds

  @functools.cached_property
  def _default_iterator(self) -> dataset.DatasetIterator:
    if isinstance(self._ds, dataset.IterDataset):
      return ElasticIterDatasetIterator(
          self._ds,
          self._global_batch_size,
          self._shard_options,
          read_options=self._read_options,
      )
    else:
      return _ElasticMapDatasetIterator(
          self._ds,
          self._global_batch_size,
          self._shard_options,
          read_options=self._read_options,
          multiprocessing_options=self._multiprocessing_options,
      )

  @property
  def shard_options(self) -> sharding.ShardOptions:
    return self._shard_options

  def __next__(self) -> Any:
    """For backwards compatibility with direct next() calls.

    This allows using ElasticIterator as a one-shot iterator directly,
    delegating to a default iterator created on the first call.

    Returns:
      The next element in the iteration.
    """
    return next(self._default_iterator)

  def close(self) -> None:
    """Closes the default iterator if it was created for backwards compatibility."""
    if "_default_iterator" in self.__dict__:
      self._default_iterator.close()

  def get_state(self) -> dict[str, Any]:
    """Returns the state of the iterator."""
    return self._default_iterator.get_state()

  def set_state(self, state):
    """Sets the state of the iterator.

    Args:
      state: The state to set.
    """
    if "_default_iterator" in self.__dict__:
      self._default_iterator.close()
      self.__dict__.pop("_default_iterator")
    self._default_iterator.set_state(state)

  def get_shard_states(self) -> Any:
    """Returns the state of the inner iterator.

    This is only intended to be used when the ElasticIterator changes its
    topology or the number of hosts.
    """
    if not isinstance(self._default_iterator, ElasticIterDatasetIterator):
      raise NotImplementedError(
          "get_shard_states is only supported for IterDataset-based"
          " ElasticIterator."
      )
    return self._default_iterator.get_shard_states()

  def set_shard_states(self, shard_states: Any) -> None:
    """Sets the state of the inner iterator.

    This is only intended to be used when the ElasticIterator changes its
    topology or the number of hosts.

    Args:
      shard_states: The shard states to set.
    """
    if not isinstance(self._default_iterator, ElasticIterDatasetIterator):
      raise NotImplementedError(
          "set_shard_states is only supported for IterDataset-based"
          " ElasticIterator."
      )
    self._default_iterator.set_shard_states(shard_states)
