# Copyright 2026 Google LLC
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
"""Hugging Face streaming dataset wrapper."""

from collections.abc import Iterable, Iterator
import functools
from typing import Any, Protocol, runtime_checkable

from absl import logging
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats


@runtime_checkable
class HFIterableDatasetProtocol(Iterable[Any], Protocol):
  """Protocol for Hugging Face IterableDataset (streaming dataset)."""

  def __iter__(self) -> Iterator[Any]:
    ...

  def shard(
      self, num_shards: int, index: int, **kwargs
  ) -> "HFIterableDatasetProtocol":
    ...

  def state_dict(self) -> dict[str, Any]:
    ...

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    ...


class HFDatasetIterator(dataset.DatasetIterator[Any]):
  """Iterator for Hugging Face streaming datasets with state management."""

  def __init__(self, hf_ds: HFIterableDatasetProtocol):
    super().__init__()
    self._hf_ds = hf_ds
    self._count_elements_read = 0
    self._sharded = False

  @functools.cached_property
  def _hf_iter(self) -> Iterator[Any]:
    if (
        not self._sharded
        and self._ctx
        and self._ctx.mp_context.process_count > 1
    ):
      try:
        self._hf_ds = self._hf_ds.shard(
            num_shards=self._ctx.mp_context.process_count,
            index=self._ctx.mp_context.process_index,
        )
        self._sharded = True
      except (AttributeError, NotImplementedError):
        # Catch when shard() is not implemented or not supported.
        logging.log_first_n(
            logging.WARNING,
            "Hugging Face dataset doesn't support shard() method. "
            "Pipeline will run without sharding, meaning all worker processes "
            "will process the same dataset. For correct sharding, please "
            "implement shard() on your dataset.",
            1,
        )
    return iter(self._hf_ds)

  @stats.record_next_duration_if_output
  @stats.trace_input_pipeline_next(stage_category=stats.IPL_CAT_READ)
  def __next__(self) -> Any:
    val = next(self._hf_iter)
    self._count_elements_read += 1
    return val

  def get_state(self) -> dict[str, Any]:
    state = {"count_elements_read": self._count_elements_read}
    _ = self._hf_iter
    try:
      state["hf_state_dict"] = self._hf_ds.state_dict()
    except (AttributeError, NotImplementedError):
      # Catch when state_dict() is not implemented or not supported.
      pass
    return state

  def set_state(self, state: dict[str, Any]) -> None:
    self._count_elements_read = state.get("count_elements_read", 0)
    _ = self._hf_iter
    if "hf_state_dict" in state:
      try:
        self._hf_ds.load_state_dict(state["hf_state_dict"])
        self.__dict__.pop("_hf_iter", None)
        return
      except (AttributeError, NotImplementedError):
        # Catch when load_state_dict() is not implemented or not supported.
        pass
    # Fallback: recreate iterator and fast-forward.
    self.__dict__.pop("_hf_iter", None)
    for _ in range(self._count_elements_read):
      next(self._hf_iter)

  def _get_next_index(self) -> int:
    return self._count_elements_read

  def _set_next_index(self, index: int) -> None:
    self.set_state({"count_elements_read": index})


class HFIterDataset(dataset.IterDataset[Any]):
  """Wrapper for Hugging Face streaming datasets (`datasets.IterableDataset`).

  Provides native PyGrain integration for Hugging Face streaming datasets,
  supporting pipeline sharding across worker processes, dataset slicing, and
  state checkpointing. The HuggingFace dataset API already provides support for
  both remote dataset loading from HuggingFace data catalog but also local JSON
  files. To enable streaming in Hugging Face make sure to set `streaming=True`
  when calling `datasets.load_dataset()`.

  Example:
    Wrap an iterable HuggingFace dataset reading from local JSON files::
      import datasets
      import grain.python as grain

      # Use HuggingFace to read from local JSON files
      hf_dataset = datasets.load_dataset("json", "/path/to/json/files",
      streaming=True)

      # Wrap it in PyGrain's HFIterDataset
      grain_ds = grain.experimental.HFIterDataset(hf_dataset)

      # Apply PyGrain transformations
      grain_ds = grain_ds.map(transform_fn)

      # Iterate over dataset
      for example in grain_ds:
        print(example)
  """

  def __init__(self, hf_ds: HFIterableDatasetProtocol):
    super().__init__()
    if not isinstance(hf_ds, Iterable):
      raise TypeError(f"Expected an iterable object, got {type(hf_ds)}")
    self._hf_ds = hf_ds

  def __iter__(self) -> HFDatasetIterator:
    return HFDatasetIterator(self._hf_ds)

  def __str__(self) -> str:
    return "HFIterDataset"

  def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
    del sequential_slice
    num_shards = sl.step if sl.step is not None else 1
    index = sl.start if sl.start is not None else 0
    if num_shards > 1:
      try:
        self._hf_ds = self._hf_ds.shard(num_shards, index)
      except (AttributeError, NotImplementedError):
        logging.log_first_n(
            logging.WARNING,
            "Hugging Face dataset doesn't support shard() method. "
            "Pipeline will run without slicing/sharding. For correct"
            " sharding, please implement shard() on your dataset.",
            1,
        )
