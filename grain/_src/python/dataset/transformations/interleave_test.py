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

import threading
from typing import cast

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import base
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import prefetch
from grain._src.python.dataset.transformations import repeat
from grain._src.python.dataset.transformations import zip as zip_dataset
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint
import numpy as np

_INTERLEAVE_TEST_CASES = (
    dict(
        testcase_name="cycle_length_1",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=1,
        expected=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    ),
    dict(
        testcase_name="cycle_length_2",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=2,
        expected=[1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5, 5, 5],
    ),
    dict(
        testcase_name="cycle_length_3",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=3,
        expected=[1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 4, 5, 5, 5],
    ),
    dict(
        testcase_name="same_lengths",
        to_mix=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        cycle_length=3,
        expected=[1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 4],
    ),
    dict(
        testcase_name="unsorted_lengths",
        to_mix=[[1, 1, 1], [2], [3, 3, 3, 3], [4, 4]],
        cycle_length=3,
        expected=[1, 2, 3, 1, 3, 1, 4, 3, 4, 3],
    ),
    dict(
        testcase_name="large_cycle_length",
        to_mix=[[1, 1, 1], [2], [3, 3, 3, 3], [4, 4]],
        cycle_length=10,
        expected=[1, 2, 3, 4, 1, 3, 4, 1, 3, 3],
    ),
    dict(
        testcase_name="with_empty_datasets",
        to_mix=[[1, 1, 1], [], [3, 3, 3, 3], [4, 4], []],
        cycle_length=3,
        expected=[1, 3, 1, 4, 3, 1, 4, 3, 3],
    ),
    dict(
        testcase_name="five_datasets_cycle_3",
        to_mix=[[1, 2, 3], [4], [5, 6], [7], [8, 9]],
        cycle_length=3,
        expected=[1, 4, 5, 2, 6, 3, 7, 8, 9],
    ),
)


class _IteratorIdDatasetIterator(dataset.DatasetIterator):
  """Iterator that returns its object id."""

  def __next__(self):
    return id(self)

  def get_state(self):
    return self._parent.get_state()

  def set_state(self, state):
    self._parent.set_state(state)


class _IteratorIdIterDataset(dataset.IterDataset):
  """Dataset that returns its iterator's object id."""

  def __iter__(self) -> dataset.DatasetIterator:
    return _IteratorIdDatasetIterator(self._parent.__iter__())


@absltest.skipThisClass("Base class")
class _InterleaveIterDatasetTestBase(parameterized.TestCase):

  def _create_dataset(self, *args, **kwargs):
    return interleave.InterleaveIterDataset(*args, **kwargs)

  def _maybe_wrap_ds(self, ds):
    return ds

  @parameterized.named_parameters(*_INTERLEAVE_TEST_CASES)
  def test_interleaved_mix(self, to_mix, cycle_length, expected):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in to_mix
    ]
    ds = self._create_dataset(datasets, cycle_length=cycle_length)
    ds = self._maybe_wrap_ds(ds)
    self.assertEqual(list(ds), expected)
    # Sanity check.
    flat_inputs = []
    for ds in datasets:
      flat_inputs.extend(list(ds))
    self.assertCountEqual(flat_inputs, expected)

  @parameterized.named_parameters(*_INTERLEAVE_TEST_CASES)
  def test_checkpoint(self, to_mix, cycle_length, expected):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in to_mix
    ]
    ds = self._create_dataset(datasets, cycle_length=cycle_length)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    checkpoints = {}
    for i in range(len(expected)):
      checkpoints[i] = ds_iter.get_state()
      _ = next(ds_iter)
    for i, state in checkpoints.items():
      ds_iter.set_state(state)
      self.assertEqual(
          list(ds_iter), expected[i:], msg=f"Failed at checkpoint {i}."
      )

  @parameterized.named_parameters(*_INTERLEAVE_TEST_CASES)
  def test_checkpoint_with_extra_threads_creating_iterators(
      self, to_mix, cycle_length, expected
  ):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in to_mix
    ]
    ds = self._create_dataset(
        datasets,
        cycle_length=cycle_length,
        num_make_iter_threads=10,
        make_iter_buffer_size=10,
    )
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    checkpoints = {}
    for i in range(len(expected)):
      checkpoints[i] = ds_iter.get_state()
      _ = next(ds_iter)
    for i, state in checkpoints.items():
      ds_iter.set_state(state)
      self.assertEqual(
          list(ds_iter), expected[i:], msg=f"Failed at checkpoint {i}."
      )

  def test_with_map_dataset_of_datasets(self):

    def make_dummy_source(filename):
      chars = [c for c in filename]
      return dataset.MapDataset.source(chars)

    filenames = dataset.MapDataset.source(["11", "2345", "678", "9999"])
    sources = filenames.shuffle(seed=42).map(make_dummy_source)
    ds = self._create_dataset(sources, cycle_length=2)
    ds = self._maybe_wrap_ds(ds)
    self.assertEqual(
        list(ds),
        ["1", "2", "1", "3", "4", "6", "5", "7", "8", "9", "9", "9", "9"],
    )

  def test_with_mp_prefetch(self):
    ds = dataset.MapDataset.range(1, 6).map(
        lambda i: dataset.MapDataset.source([i]).repeat(i).to_iter_dataset()
    )
    ds = self._create_dataset(ds, cycle_length=5)
    ds = self._maybe_wrap_ds(ds)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=3))
    self.assertEqual(list(ds), [1, 2, 3, 4, 5, 3, 4, 2, 3, 4, 5, 4, 5, 5, 5])

  def test_options_propagated(self):
    ds1 = dataset.MapDataset.source([1]).repeat(1000).to_iter_dataset()
    ds1 = ds1.filter(lambda x: False)
    ds2 = dataset.MapDataset.source([2]).repeat(1000).to_iter_dataset()
    ds = self._create_dataset([ds1, ds2], cycle_length=1)
    ds = self._maybe_wrap_ds(ds)
    ds_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds = dataset.WithOptionsIterDataset(ds, ds_options)
    with self.assertRaisesRegex(ValueError, r"skipped 100\.00 %"):
      list(ds)

  def test_checkpointing_comprehensive(self):
    ds = [
        dataset.MapDataset.source([i]).repeat(i).to_iter_dataset()
        for i in range(1, 6)
    ]
    ds = self._create_dataset(ds, cycle_length=5)
    ds = self._maybe_wrap_ds(ds)
    assert_equal_output_after_checkpoint(ds)

  def test_set_state_does_not_recreate_iterators_if_not_needed(self):
    cycle_length = 5
    ds = dataset.MapDataset.range(100).to_iter_dataset()
    ds = _IteratorIdIterDataset(ds)
    ds = self._create_dataset([ds] * cycle_length, cycle_length=cycle_length)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    iter_ids1 = []
    for _ in range(cycle_length):
      iter_ids1.append(next(ds_iter))
    checkpoint = ds_iter.get_state()
    next(ds_iter)
    ds_iter.set_state(checkpoint)
    iter_ids2 = []
    for _ in range(cycle_length):
      iter_ids2.append(next(ds_iter))
    self.assertEqual(iter_ids1, iter_ids2)

  def test_element_spec(self):
    ds = dataset.MapDataset.range(3).to_iter_dataset()
    ds = self._create_dataset([ds, ds], cycle_length=2)
    ds = self._maybe_wrap_ds(ds)
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())

  @flagsaver.flagsaver(grain_py_debug_mode=True)
  def test_interleave_stats(self):
    ds = dataset.MapDataset.range(10000).map(lambda x: x + 1)
    ds = ds.to_iter_dataset()
    ds = self._create_dataset([ds, ds], cycle_length=2)
    it = ds.__iter__()
    next(it)
    next(it)
    summary = dataset.get_execution_summary(it)
    node_names = {node.name for node in summary.nodes.values()}
    expected_nodes = [
        "RangeMapDataset",
        "MapMapDataset",
        "PrefetchDatasetIterator",
        "ThreadPrefetchDatasetIterator",
        "Interleave",
    ]
    for expected_node in expected_nodes:
      self.assertTrue(any(expected_node in name for name in node_names))
    self.assertLen(node_names, len(expected_nodes))

  @flagsaver.flagsaver(grain_py_debug_mode=True)
  def test_interleave_stats_with_mismatched_dataset_structures(self):
    ds1 = dataset.MapDataset.range(10000).map(lambda x: x + 1)
    ds1 = ds1.to_iter_dataset()
    ds2 = dataset.MapDataset.range(10000).map(lambda x: x + 1).map(lambda x: x)
    ds2 = ds2.to_iter_dataset()
    ds = self._create_dataset([ds1, ds2], cycle_length=2)
    it = ds.__iter__()
    next(it)
    next(it)
    summary = dataset.get_execution_summary(it)
    node_names = [node.name for node in summary.nodes.values()]
    self.assertLen(node_names, 1)
    self.assertIn("Interleave", node_names[0])

  def test_get_next_index(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    ds = interleave.InterleaveIterDataset([ds], cycle_length=1)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    self.assertEqual(dataset.get_next_index(ds_iter), 0)
    for i in range(10):
      next(ds_iter)
      self.assertEqual(dataset.get_next_index(ds_iter), i + 1)

  def test_set_next_index(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    ds = interleave.InterleaveIterDataset([ds], cycle_length=1)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    for i in reversed(range(10)):
      dataset.set_next_index(ds_iter, i)
      self.assertEqual(next(ds_iter), i)

  def test_get_next_index_with_multiple_datasets(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    ds = interleave.InterleaveIterDataset([ds, ds], cycle_length=2)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    with self.assertRaisesRegex(
        NotImplementedError,
        "get_next_index is not supported for .*Interleave",
    ):
      dataset.get_next_index(ds_iter)

  def test_set_next_index_with_multiple_datasets(self):
    ds = dataset.MapDataset.range(10).to_iter_dataset()
    ds = interleave.InterleaveIterDataset([ds, ds], cycle_length=2)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()
    with self.assertRaisesRegex(
        NotImplementedError,
        "set_next_index is not supported for .*Interleave",
    ):
      dataset.set_next_index(ds_iter, 0)

  def test_future_states(self):
    datasets = [
        dataset.MapDataset.source([1, 2]).to_iter_dataset(),
        dataset.MapDataset.source([3, 4]).to_iter_dataset(),
    ]
    ds = self._create_dataset(datasets, cycle_length=1)
    ds = self._maybe_wrap_ds(ds)
    ds_iter = ds.__iter__()

    # Initialize the first iterator and get state.
    state = ds_iter.get_state()

    # Get state for the second dataset iterator after advancing it.
    ds1_iter = datasets[1].__iter__()
    next(ds1_iter)  # Consumes 3
    ds1_state = ds1_iter.get_state()

    # Inject future state for the second dataset (index 1).
    state["future_states"] = {1: ds1_state}

    ds_iter.set_state(state)

    # Consume elements.
    # It should yield elements from the first dataset (1, 2) and then
    # yield elements from the second dataset starting from the future state (4).
    self.assertEqual(next(ds_iter), 1)
    self.assertEqual(next(ds_iter), 2)
    self.assertEqual(next(ds_iter), 4)

    with self.assertRaises(StopIteration):
      next(ds_iter)

  @parameterized.named_parameters(
      dict(
          testcase_name="cycle_length_equals_num_datasets",
          ds_elements=[[1, 2, 3], [4, 5, 6]],
          cycle_length=2,
          expected_shard_state=[
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 1}},
          ],
          expected_remaining=[2, 5, 3, 6],
      ),
      dict(
          testcase_name="cycle_length_less_than_num_datasets",
          ds_elements=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          cycle_length=2,
          expected_shard_state=[
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 0}},
          ],
          expected_remaining=[2, 5, 3, 6, 7, 8, 9],
      ),
  )
  def test_slice_state_management_checkpoints_correctly(
      self,
      ds_elements,
      cycle_length,
      expected_shard_state,
      expected_remaining,
  ):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in ds_elements
    ]
    ds = self._create_dataset(datasets, cycle_length=cycle_length)
    ds = self._maybe_wrap_ds(ds)
    it = ds.__iter__()

    # Consume some elements to advance state.
    for _ in range(2):
      next(it)

    # Get the shard state.
    assert isinstance(it, prefetch.SupportsSlicedStateManagement)
    shard_state = it.get_shard_states()
    self.assertEqual(shard_state, expected_shard_state)

    # Create a new iterator and restore state.
    it2 = ds.__iter__()
    assert isinstance(it2, prefetch.SupportsSlicedStateManagement)
    it2.set_shard_states(shard_state)

    # Verify it continues from the correct position.
    self.assertSequenceEqual(list(it2), expected_remaining)

  @parameterized.named_parameters(
      dict(
          testcase_name="cycle_length_equals_num_datasets",
          ds_elements=[[1, 2, 3], [4, 5, 6]],
          cycle_length=2,
          expected_shard_state=[
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 1}},
          ],
          expected_future_states={},
          expected_next_index_in_datasets=2,
          expected_iterators_in_use_indices=[0, 1],
          expected_iterators_in_use_states=[
              {"next_index": 1},
              {"next_index": 1},
          ],
      ),
      dict(
          testcase_name="cycle_length_less_than_num_datasets",
          ds_elements=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
          cycle_length=2,
          expected_shard_state=[
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 1}},
              {"exhausted": 0, "state": {"next_index": 0}},
          ],
          expected_future_states={2: {"next_index": 0}},
          expected_next_index_in_datasets=2,
          expected_iterators_in_use_indices=[0, 1],
          expected_iterators_in_use_states=[
              {"next_index": 1},
              {"next_index": 1},
          ],
      ),
  )
  def test_correct_interleave_state_after_setting_shards(
      self,
      ds_elements,
      cycle_length,
      expected_shard_state,
      expected_future_states,
      expected_next_index_in_datasets,
      expected_iterators_in_use_indices,
      expected_iterators_in_use_states,
  ):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in ds_elements
    ]
    ds = self._create_dataset(datasets, cycle_length=cycle_length)
    ds = self._maybe_wrap_ds(ds)
    it = ds.__iter__()

    # Consume some elements to advance state.
    for _ in range(2):
      next(it)

    assert isinstance(it, prefetch.SupportsSlicedStateManagement)
    shard_state = it.get_shard_states()
    self.assertEqual(shard_state, expected_shard_state)

    # Create a new iterator and restore state.
    it2 = ds.__iter__()
    assert isinstance(it2, prefetch.SupportsSlicedStateManagement)
    it2.set_shard_states(shard_state)

    # Check get_shard_states() returns the set shard states correctly.
    self.assertEqual(it2.get_shard_states(), expected_shard_state)

    # Check get_state() internal values.
    state = it2.get_state()
    self.assertEqual(state["next_index_in_cycle"], 0)
    self.assertEqual(
        state["next_index_in_datasets"], expected_next_index_in_datasets
    )
    self.assertEqual(
        state["iterators_in_use_indices"], expected_iterators_in_use_indices
    )
    self.assertEqual(
        state["iterators_in_use_states"], expected_iterators_in_use_states
    )
    self.assertEqual(state["future_states"], expected_future_states)

  def test_setting_shard_state_with_exhausted_states(self):
    datasets_data = [[1], [2, 3], [4, 5]]
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in datasets_data
    ]
    ds = self._create_dataset(datasets, cycle_length=2)
    ds = self._maybe_wrap_ds(ds)
    it = ds.__iter__()

    shard_state = [
        {"exhausted": 1, "state": {"next_index": 1}},
        {"exhausted": 1, "state": {"next_index": 2}},
        {"exhausted": 0, "state": {"next_index": 0}},
    ]

    # Create a new iterator and restore state.
    assert isinstance(it, prefetch.SupportsSlicedStateManagement)
    it.set_shard_states(shard_state)

    # Check get_state() internal values.
    state = it.get_state()
    self.assertEqual(state["next_index_in_cycle"], 0)
    self.assertEqual(state["next_index_in_datasets"], 3)
    self.assertEqual(state["iterators_in_use_indices"], [2, 0])
    self.assertEqual(
        state["iterators_in_use_states"],
        [{"next_index": 0}, {"next_index": 1}],
    )
    if isinstance(self, InterleaveIterDatasetTest):
      self.assertEqual(state["exhausted"], [0, 1])

  def test_options_propagated_with_interleaved_interleaves(self):
    ds = (
        dataset.MapDataset.range(0, 1500)
        .to_iter_dataset()
        .filter(lambda x: False)
    )
    interleave_ds = self._create_dataset([ds], cycle_length=1)
    interleave_ds_2 = self._create_dataset([interleave_ds], cycle_length=1)

    filter_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds_with_options = dataset.WithOptionsIterDataset(
        interleave_ds_2, filter_options
    )
    with self.assertRaisesRegex(ValueError, r"skipped 100\.00 %"):
      list(ds_with_options)

  def test_options_propagated_with_zipped_interleaves(self):
    no_filter_ds = dataset.MapDataset.range(
        1200, 1500
    ).to_iter_dataset()  # 300 elements

    filter_ds = (
        dataset.MapDataset.range(0, 1500)
        .to_iter_dataset()
        .filter(lambda x: x >= 1200)
    )
    interleave_ds1 = self._create_dataset([filter_ds], cycle_length=1)
    interleave_ds2 = self._create_dataset([no_filter_ds], cycle_length=1)
    zipped_ds = zip_dataset.ZipIterDataset([interleave_ds1, interleave_ds2])
    zipped_ds2 = zip_dataset.ZipIterDataset([interleave_ds2, interleave_ds1])

    filter_options = base.DatasetOptions(filter_raise_threshold_ratio=0.1)
    ds_with_options1 = dataset.WithOptionsIterDataset(zipped_ds, filter_options)
    ds_with_options2 = dataset.WithOptionsIterDataset(
        zipped_ds2, filter_options
    )

    with self.assertRaisesRegex(
        ValueError, r"FilterDatasetIterator.*skipped 100\.00 %"
    ):
      list(ds_with_options1)

    with self.assertRaisesRegex(
        ValueError, r"FilterDatasetIterator.*skipped 100\.00 %"
    ):
      list(ds_with_options2)


if __name__ == "__main__":
  absltest.main()
