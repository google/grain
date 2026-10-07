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
"""Tests for HFIterDataset wrapper."""

from typing import Any, Iterator
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import base
from grain._src.python.dataset.sources import hf_source


class DummyStreamingDataset:
  """Dummy streaming dataset simulating Hugging Face IterableDataset."""

  def __init__(
      self, data: list[Any], shard_info: tuple[int, int] | None = None
  ):
    self.data = data
    self.shard_info = shard_info
    self.shard_called_with = None
    self._state = 0

  def __iter__(self) -> Iterator[Any]:
    for item in self.data:
      yield item

  def shard(self, num_shards: int, index: int, **_) -> "DummyStreamingDataset":
    self.shard_called_with = (num_shards, index)
    sharded_data = [
        item for i, item in enumerate(self.data) if i % num_shards == index
    ]
    return DummyStreamingDataset(sharded_data, (num_shards, index))

  def state_dict(self) -> dict[str, Any]:
    raise NotImplementedError

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    raise NotImplementedError


class DummyDatasetWithStateDict(DummyStreamingDataset):
  """Dummy dataset simulating Hugging Face IterableDataset with state_dict."""

  def __init__(self, data: list[Any], start_idx: int = 0):
    super().__init__(data)
    self.start_idx = start_idx
    self._current_idx = start_idx

  def __iter__(self) -> Iterator[Any]:
    for idx in range(self.start_idx, len(self.data)):
      self._current_idx = idx + 1
      yield self.data[idx]

  def state_dict(self) -> dict[str, Any]:
    return {"idx": self._current_idx}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self.start_idx = state_dict["idx"]
    self._current_idx = self.start_idx


class HFIterDatasetTest(parameterized.TestCase):

  def test_protocol_conformance(self):
    dummy_ds = DummyStreamingDataset([1, 2, 3])
    self.assertIsInstance(dummy_ds, hf_source.HFIterableDatasetProtocol)

  def test_basic_iteration(self):
    data = [{"id": i} for i in range(5)]
    dummy_ds = DummyStreamingDataset(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)
    self.assertEqual(list(grain_ds), data)

  def test_multiprocess_sharding(self):
    data = list(range(10))
    dummy_ds = DummyStreamingDataset(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)

    # Simulate worker 1 of 2
    ctx = base.IteratorContext(
        mp_context=base.MultiprocessingContext(process_index=1, process_count=2)
    )
    # Inject context into iterator
    iterator = grain_ds.__iter__()
    iterator._ctx = ctx
    self.assertEqual(list(iterator), [1, 3, 5, 7, 9])

  def test_checkpointing_fallback_fast_forward(self):
    data = list(range(10))
    dummy_ds = DummyStreamingDataset(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)
    iterator = grain_ds.__iter__()

    self.assertEqual(next(iterator), 0)
    self.assertEqual(next(iterator), 1)

    state = iterator.get_state()
    self.assertEqual(state, {"count_elements_read": 2})

    # Read more elements
    self.assertEqual(next(iterator), 2)
    self.assertEqual(next(iterator), 3)

    # Restore state
    iterator.set_state(state)
    self.assertEqual(next(iterator), 2)

  def test_checkpointing_with_dataset_state_dict(self):
    data = list(range(10))
    dummy_ds = DummyDatasetWithStateDict(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)
    iterator = grain_ds.__iter__()

    self.assertEqual(next(iterator), 0)
    self.assertEqual(next(iterator), 1)

    state = iterator.get_state()
    self.assertEqual(state["count_elements_read"], 2)
    self.assertEqual(state["hf_state_dict"], {"idx": 2})

    # Read more elements
    self.assertEqual(next(iterator), 2)

    # Restore state
    iterator.set_state(state)
    self.assertEqual(next(iterator), 2)

  def test_multiprocess_sharding_checkpointing_with_state_dict(self):
    data = list(range(10))
    dummy_ds = DummyDatasetWithStateDict(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)

    ctx = base.IteratorContext(
        mp_context=base.MultiprocessingContext(process_index=1, process_count=2)
    )
    iterator = grain_ds.__iter__()
    iterator._ctx = ctx

    # Shard index 1 of 2 yields: 1, 3, 5, 7, 9
    self.assertEqual(next(iterator), 1)
    self.assertEqual(next(iterator), 3)

    state = iterator.get_state()
    self.assertEqual(next(iterator), 5)

    # Restore state
    iterator.set_state(state)
    self.assertEqual(next(iterator), 5)

  def test_set_slice(self):
    data = list(range(10))
    dummy_ds = DummyStreamingDataset(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)

    # Slice the dataset (worker index 1 of 2 workers)
    grain_ds.set_slice(slice(1, None, 2))

    # Verify underlying dataset object was sharded
    self.assertEqual(getattr(grain_ds._hf_ds, "shard_info"), (2, 1))

    # Iterate and verify it yields the sliced data
    self.assertEqual(list(grain_ds), [1, 3, 5, 7, 9])

  def test_set_slice_unsupported_sharding(self):
    class UnshardableDataset:

      def __init__(self, data: list[Any]):
        self.data = data

      def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    dummy_ds = UnshardableDataset(list(range(5)))
    grain_ds = hf_source.HFIterDataset(dummy_ds)  # pytype: disable=wrong-arg-types
    # Should not raise AttributeError when set_slice is called.
    grain_ds.set_slice(slice(1, None, 2))
    self.assertEqual(list(grain_ds), list(range(5)))

  def test_next_index_methods(self):
    data = list(range(5))
    dummy_ds = DummyStreamingDataset(data)
    grain_ds = hf_source.HFIterDataset(dummy_ds)
    iterator = grain_ds.__iter__()

    self.assertEqual(iterator._get_next_index(), 0)
    next(iterator)
    self.assertEqual(iterator._get_next_index(), 1)
    iterator._set_next_index(3)
    self.assertEqual(next(iterator), 3)


if __name__ == "__main__":
  absltest.main()
