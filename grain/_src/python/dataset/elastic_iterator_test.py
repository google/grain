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
import contextlib
import itertools
import platform
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from grain._src.core import sharding
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.checkpoint import handler
from grain._src.python.dataset import dataset
from grain._src.python.dataset import elastic_iterator
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import zip as zip_transform
import grain._src.python.testing.experimental as test_util
import numpy as np


def assert_equal_bag_output_after_checkpoint(ds_iter):
  checkpoints = []
  expected_values = []
  with contextlib.closing(ds_iter) as iterator:
    for _ in itertools.count():
      current_state = iterator.get_state()
      try:
        value = next(iterator)
      except StopIteration:
        break
      checkpoints.append(current_state)
      expected_values.append(value)

  assert expected_values, "Dataset did not produce any elements."

  for i, state in enumerate(checkpoints):
    with contextlib.closing(ds_iter) as new_iterator:
      new_iterator.set_state(state)
      new_values = list(new_iterator)

      np.testing.assert_equal(
          sorted(new_values),
          sorted(expected_values[i:]),
          f"Restored values mismatch (ignoring order) at step {i} for state"
          f" {state}.",
      )


@absltest.skipIf(platform.system() == "Windows", "Skipped under bazel.")
class ElasticMapDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          global_batch_size=5,
          shard_options=sharding.NoSharding(),
          multiprocessing_options=None,
          expected=[np.arange(1, 6), np.arange(6, 11)],
      ),
      dict(
          global_batch_size=2,
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
          multiprocessing_options=None,
          expected=[[1], [3], [5], [7], [9]],
      ),
      dict(
          global_batch_size=8,
          shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
          multiprocessing_options=None,
          expected=[[3, 7]],
      ),
      dict(
          global_batch_size=2,
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
          multiprocessing_options=options.MultiprocessingOptions(num_workers=7),
          expected=[[1], [3], [5], [7], [9]],
      ),
  )
  def test_produces_correct_elements(
      self, global_batch_size, shard_options, multiprocessing_options, expected
  ):
    ds = dataset.MapDataset.range(10).map(lambda x: x + 1)
    actual = list(
        elastic_iterator.ElasticIterator(
            ds,
            global_batch_size,
            shard_options,
            multiprocessing_options=multiprocessing_options,
        )
    )
    np.testing.assert_equal(
        actual, expected, err_msg=f"actual: {actual}, expected: {expected}"
    )

  def test_checkpointing(self):
    ds = dataset.MapDataset.range(100).map(lambda x: x * 2).shuffle(42)
    it = elastic_iterator.ElasticIterator(ds, 5, sharding.NoSharding())
    test_util.assert_equal_output_after_checkpoint(it)

  def test_checkpointing_with_multiprocessing(self):
    ds = dataset.MapDataset.range(5).map(lambda x: x * 2).shuffle(42)
    it = elastic_iterator.ElasticIterator(
        ds,
        2,
        sharding.NoSharding(),
        multiprocessing_options=options.MultiprocessingOptions(2),
    )
    test_util.assert_equal_output_after_checkpoint(it)

  def _elastic_resize_test_base(
      self, make_iterators_before, make_iterators_after, all_expected_elements
  ):
    iterators = make_iterators_before()
    # Advance all iterators by 7 steps.
    all_actual_elements = set()
    for it in iterators:
      for _ in range(7):
        element = next(it)
        all_actual_elements.update(element.tolist())
    # Get state of any iterator.
    state = iterators[0].get_state()
    iterators = make_iterators_after()
    # Restore state of all iterators.
    for it in iterators:
      it.set_state(state)
    # Advance all iterators by the remaining steps.
    for it in iterators:
      for element in it:
        all_actual_elements.update(element.tolist())

    # Check that all elements were produced exactly once.
    self.assertSetEqual(all_actual_elements, all_expected_elements)

  def test_elastic_downsize(self):
    ds = dataset.MapDataset.range(1024).map(lambda x: x * 2).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))

    # Create iterators over 32 hosts with per-host batch size 2.
    def make_iterators_before():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              64,
              sharding.ShardOptions(shard_index=i, shard_count=32),
          )
          for i in range(32)
      ]

    # Create new iterators over 16 hosts with per-host batch size 2.
    def make_iterators_after():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              32,
              sharding.ShardOptions(shard_index=i, shard_count=16),
          )
          for i in range(16)
      ]

    self._elastic_resize_test_base(
        make_iterators_before, make_iterators_after, all_expected_elements
    )

  def test_elastic_downsize_with_multiprocessing(self):
    ds = dataset.MapDataset.range(2**16).map(lambda x: x * 2).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))

    # Create iterators over 8 hosts with per-host batch size 32.
    def make_iterators_before():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              256,
              sharding.ShardOptions(shard_index=i, shard_count=8),
              multiprocessing_options=options.MultiprocessingOptions(
                  num_workers=2
              ),
          )
          for i in range(8)
      ]

    # Create new iterators over 4 hosts with per-host batch size 32.
    def make_iterators_after():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              128,
              sharding.ShardOptions(shard_index=i, shard_count=4),
              multiprocessing_options=options.MultiprocessingOptions(
                  num_workers=2
              ),
          )
          for i in range(4)
      ]

    self._elastic_resize_test_base(
        make_iterators_before, make_iterators_after, all_expected_elements
    )

  def test_elastic_upsize(self):
    ds = dataset.MapDataset.range(2**16).map(lambda x: x - 1).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))

    # Create iterators over 8 hosts with per-host batch size 16.
    def make_iterators_before():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              128,
              sharding.ShardOptions(shard_index=i, shard_count=8),
          )
          for i in range(8)
      ]

    # Create new iterators over 64 hosts with per-host batch size 2.
    def make_iterators_after():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              128,
              sharding.ShardOptions(shard_index=i, shard_count=64),
          )
          for i in range(64)
      ]

    self._elastic_resize_test_base(
        make_iterators_before, make_iterators_after, all_expected_elements
    )

  def test_elastic_upsize_with_multiprocessing(self):
    ds = dataset.MapDataset.range(2**16).map(lambda x: x - 1).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))

    # Create iterators over 4 hosts with per-host batch size 16.
    def make_iterators_before():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              64,
              sharding.ShardOptions(shard_index=i, shard_count=4),
              multiprocessing_options=options.MultiprocessingOptions(
                  num_workers=2
              ),
          )
          for i in range(4)
      ]

    # Create new iterators over 6 hosts with per-host batch size 16.
    def make_iterators_after():
      return [
          elastic_iterator.ElasticIterator(
              ds,
              96,
              sharding.ShardOptions(shard_index=i, shard_count=6),
              multiprocessing_options=options.MultiprocessingOptions(
                  num_workers=2
              ),
          )
          for i in range(6)
      ]

    self._elastic_resize_test_base(
        make_iterators_before, make_iterators_after, all_expected_elements
    )

  def test_filter_raises_error(self):
    ds = dataset.MapDataset.range(10).map(lambda x: x + 1)
    ds = ds.filter(lambda x: x % 2 == 0)
    with self.assertRaisesRegex(
        ValueError,
        "ElasticIterator does not support `filter` transformation.",
    ):
      elastic_iterator.ElasticIterator(ds, 5, sharding.NoSharding()).__iter__()

  def test_shard_state_not_implemented(self):
    ds = dataset.MapDataset.range(10).map(lambda x: x + 1)
    it = elastic_iterator.ElasticIterator(ds, 5, sharding.NoSharding())
    with self.assertRaisesRegex(
        NotImplementedError, "get_shard_states is only supported"
    ):
      it.get_shard_states()
    with self.assertRaisesRegex(
        NotImplementedError, "set_shard_states is only supported"
    ):
      it.set_shard_states({})


class ElasticIterDatasetTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          shard_options=sharding.NoSharding(),
          global_batch_size=1,
          expected=list(range(15)),
      ),
      dict(
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=1),
          global_batch_size=1,
          expected=list(range(15)),
      ),
      dict(
          shard_options=sharding.NoSharding(),
          global_batch_size=3,
          # Data is interleaved with cycle length 3.
          expected=[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]],
      ),
  )
  def test_no_sharding_produces_correct_elements(
      self, shard_options, global_batch_size, expected
  ):
    ds = [
        # 3 shards, each with 5 elements.
        dataset.MapDataset.range(i * 5, (i + 1) * 5).to_iter_dataset()
        for i in range(3)
    ]
    interleave_ds = interleave.InterleaveIterDataset(
        ds, cycle_length=global_batch_size
    )
    it = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=shard_options,
        global_batch_size=global_batch_size,
    )
    actual = list(it)
    np.testing.assert_equal(actual, expected)

  @parameterized.parameters(
      dict(
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
          global_batch_size=2,
          expected=[[0], [2], [4], [6], [8]],
      ),
      dict(
          shard_options=sharding.ShardOptions(shard_index=1, shard_count=2),
          global_batch_size=0,
          expected=[1, 3, 5, 7, 9],
      ),
      dict(
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
          global_batch_size=4,
          expected=[[0, 2], [4, 6], [8]],
      ),
  )
  def test_sharding_produces_correct_elements(
      self, shard_options, global_batch_size, expected
  ):
    ds = [
        # 4 shards, 0: [0, 4, 8], 1: [1, 5, 9], 2: [2, 6], 3: [3, 7]
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset()
        for i in range(4)
    ]
    # Use cycle_length=2 as in the original test.
    interleave_ds = interleave.InterleaveIterDataset(ds, cycle_length=2)
    it = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=shard_options,
        global_batch_size=global_batch_size,
    )
    actual = list(it)
    np.testing.assert_equal(actual, expected)

  def test_zip_sharding_produces_correct_elements(self):
    """Tests elastic sharding for ZipIterDataset over stream/sequential iterators (InterleaveIterDataset)."""
    ds1 = [
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset() for i in range(4)
    ]
    interleave_ds1 = interleave.InterleaveIterDataset(ds1, cycle_length=2)

    ds2 = [
        dataset.MapDataset.range(i + 10, 20, 4).to_iter_dataset()
        for i in range(4)
    ]
    interleave_ds2 = interleave.InterleaveIterDataset(ds2, cycle_length=2)

    zip_ds = zip_transform.ZipIterDataset([interleave_ds1, interleave_ds2])

    it = elastic_iterator.ElasticIterator(
        zip_ds,
        shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
        global_batch_size=2,
    )
    actual = list(it)
    # Zip of Batches: Tuple of Lists
    expected = [
        ([0], [10]),
        ([2], [12]),
        ([4], [14]),
        ([6], [16]),
        ([8], [18]),
    ]
    self.assertEqual(actual, expected)

  def test_checkpointing_no_change(self):
    ds = [
        dataset.MapDataset.range(i, 100, 25).to_iter_dataset()
        for i in range(25)
    ]
    global_batch_size = 4
    interleave_ds = interleave.InterleaveIterDataset(
        ds, cycle_length=global_batch_size
    )
    it = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    assert_equal_bag_output_after_checkpoint(it)

  def test_checkpointing_with_map_iter_dataset(self):
    ds = [
        dataset.MapDataset.range(i, 100, 25).to_iter_dataset()
        for i in range(25)
    ]
    global_batch_size = 4
    interleave_ds = interleave.InterleaveIterDataset(
        ds, cycle_length=global_batch_size
    )
    mapped_ds = interleave_ds.map(lambda x: x + 1)
    it = elastic_iterator.ElasticIterator(
        mapped_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    assert_equal_bag_output_after_checkpoint(it)

  def test_get_set_state(self):
    ds = [
        dataset.MapDataset.range(i, 100, 25).to_iter_dataset()
        for i in range(25)
    ]
    global_batch_size = 4
    interleave_ds = interleave.InterleaveIterDataset(
        ds, cycle_length=global_batch_size
    )
    it = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    _ = [next(it) for _ in range(5)]

    inner_state = it.get_state()

    it2 = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    it2.set_state(inner_state)

    self.assertEqual(next(it), next(it2))

  def test_set_shard_states_with_string_keys(self):
    ds = [
        dataset.MapDataset.range(i, 100, 25).to_iter_dataset()
        for i in range(25)
    ]
    global_batch_size = 4
    interleave_ds = interleave.InterleaveIterDataset(
        ds, cycle_length=global_batch_size
    )
    it = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    _ = [next(it) for _ in range(4)]

    shard_states = it.get_shard_states()

    # Simulate JSON serialization/deserialization.
    # Convert keys in the sharded state to strings to simulate JSON behavior.
    json_like_shard_states = {str(k): v for k, v in shard_states.items()}

    it2 = elastic_iterator.ElasticIterator(
        interleave_ds,
        shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
        global_batch_size=global_batch_size,
    )
    # This should not raise an error even though keys are strings.
    it2.set_shard_states(json_like_shard_states)

    self.assertEqual(next(it), next(it2))

  def _create_sharded_datasource(
      self,
      cycle_length=10,
  ):
    # Dataset looks like:
    # [0, 25, 50, 75]
    # [1, 26, 51, 76]
    # [2, 27, 52, 77]
    # ...
    # [24, 49, 74, 99]
    datasource_ds = [
        dataset.MapDataset.range(i, 100, 25).to_iter_dataset()
        for i in range(25)
    ]
    interleave_ds = interleave.InterleaveIterDataset(
        datasource_ds, cycle_length=cycle_length
    )
    return interleave_ds

  def _create_iterators(
      self,
      ds: dataset.MapDataset | dataset.IterDataset,
      shard_count: int,
      global_batch_size: int,
  ) -> list[elastic_iterator.ElasticIterator]:
    return [
        elastic_iterator.ElasticIterator(
            ds,
            shard_options=sharding.ShardOptions(
                shard_index=i, shard_count=shard_count
            ),
            global_batch_size=global_batch_size,
        )
        for i in range(shard_count)
    ]

  def _consume_elements(
      self,
      iterators: list[elastic_iterator.ElasticIterator],
      num_elements: int,
  ) -> list[Any]:
    actual_elements = []
    for i in range(num_elements):
      actual_elements.append(next(iterators[i % len(iterators)]))
    return actual_elements

  def _consume_remaining(
      self,
      iterators: list[elastic_iterator.ElasticIterator],
  ) -> list[Any]:
    actual_elements = []
    iterators = list(iterators)
    i = 0
    while iterators:
      it_index = i % len(iterators)
      try:
        element = next(iterators[it_index])
      except StopIteration:
        iterators[it_index].close()
        iterators.pop(it_index)
        continue
      actual_elements.append(element)
      i += 1
    return actual_elements

  def _flatten_and_assert_equal(
      self,
      actual_elements: list[Any],
      expected_elements: list[Any],
  ):
    flat_elements = []
    for batch in actual_elements:
      flat_elements.extend(batch)
    self.assertCountEqual(flat_elements, expected_elements)

  def _save_elastic_iterators(
      self,
      directory: str,
      iterators: list[elastic_iterator.ElasticIterator],
  ):
    directory = epath.Path(directory)
    checkpoint_handler = handler.CheckpointHandler()
    for i, iterator in enumerate(iterators):
      with mock.patch.object(
          sharding,
          "get_process_index_and_count",
          return_value=(i, len(iterators)),
      ):
        checkpoint_handler.save(directory, iterator)

  def _restore_elastic_iterators(
      self,
      directory: str,
      iterators: list[elastic_iterator.ElasticIterator],
  ):
    directory = epath.Path(directory)
    checkpoint_handler = handler.CheckpointHandler()
    for i, iterator in enumerate(iterators):
      with mock.patch.object(
          sharding,
          "get_process_index_and_count",
          return_value=(i, len(iterators)),
      ):
        checkpoint_handler.restore(directory, iterator)

  def test_checkpointing_with_scale_up(self):
    temp_dir = self.create_tempdir()
    global_batch_size = 10
    # Dataset looks like:
    # [0, 25, 50, 75]
    # [1, 26, 51, 76]
    # [2, 27, 52, 77]
    # ...
    # [24, 49, 74, 99]
    interleave_ds = self._create_sharded_datasource()
    elastic_iterators = self._create_iterators(
        interleave_ds, 5, global_batch_size
    )
    all_elements = list(range(100))

    actual_elements = self._consume_elements(elastic_iterators, 5)

    # Save the state of the iterators all at once.
    self._save_elastic_iterators(temp_dir.full_path, elastic_iterators)

    new_elastic_iterators = self._create_iterators(
        interleave_ds, 10, global_batch_size
    )
    self._restore_elastic_iterators(temp_dir.full_path, new_elastic_iterators)

    actual_elements.extend(self._consume_remaining(new_elastic_iterators))
    self._flatten_and_assert_equal(actual_elements, all_elements)

  def test_checkpointing_with_scale_down(self):
    global_batch_size = 10

    temp_dir = self.create_tempdir()

    interleaved_ds = self._create_sharded_datasource()

    elastic_iterators = self._create_iterators(
        interleaved_ds, 10, global_batch_size
    )
    all_elements = list(range(100))

    actual_elements = self._consume_elements(elastic_iterators, 25)

    # Save the state of the iterators.
    self._save_elastic_iterators(temp_dir.full_path, elastic_iterators)

    new_elastic_iterators = self._create_iterators(
        interleaved_ds, 2, global_batch_size
    )
    self._restore_elastic_iterators(temp_dir.full_path, new_elastic_iterators)

    actual_elements.extend(self._consume_remaining(new_elastic_iterators))
    self._flatten_and_assert_equal(actual_elements, all_elements)

  def _create_zipped_datasource(self, cycle_length=10):
    ds1 = self._create_sharded_datasource(cycle_length)
    datasource_ds2 = [
        dataset.MapDataset.range(i + 100, 200, 25).to_iter_dataset()
        for i in range(25)
    ]
    ds2 = interleave.InterleaveIterDataset(
        datasource_ds2, cycle_length=cycle_length
    )
    return zip_transform.ZipIterDataset([ds1, ds2])

  def _unzip_and_flatten_zipped_elements(self, actual_elements):
    flat_x = []
    flat_y = []
    for batch in actual_elements:
      flat_x.extend(batch[0])
      flat_y.extend(batch[1])
    return flat_x, flat_y

  def test_zip_checkpointing_with_scale_down(self):
    global_batch_size = 10
    temp_dir = self.create_tempdir()
    ds = self._create_zipped_datasource()
    elastic_iterators = self._create_iterators(ds, 10, global_batch_size)
    actual_elements = self._consume_elements(elastic_iterators, 25)
    self._save_elastic_iterators(temp_dir.full_path, elastic_iterators)

    new_elastic_iterators = self._create_iterators(ds, 2, global_batch_size)
    self._restore_elastic_iterators(temp_dir.full_path, new_elastic_iterators)
    actual_elements.extend(self._consume_remaining(new_elastic_iterators))

    flat_x, flat_y = self._unzip_and_flatten_zipped_elements(actual_elements)
    self.assertCountEqual(flat_x, list(range(100)))
    self.assertCountEqual(flat_y, list(range(100, 200)))

  def test_zip_checkpointing_with_scale_up(self):
    global_batch_size = 10
    temp_dir = self.create_tempdir()
    ds = self._create_zipped_datasource()
    elastic_iterators = self._create_iterators(ds, 5, global_batch_size)
    actual_elements = self._consume_elements(elastic_iterators, 5)
    self._save_elastic_iterators(temp_dir.full_path, elastic_iterators)

    new_elastic_iterators = self._create_iterators(ds, 10, global_batch_size)
    self._restore_elastic_iterators(temp_dir.full_path, new_elastic_iterators)
    actual_elements.extend(self._consume_remaining(new_elastic_iterators))

    flat_x, flat_y = self._unzip_and_flatten_zipped_elements(actual_elements)
    self.assertCountEqual(flat_x, list(range(100)))
    self.assertCountEqual(flat_y, list(range(100, 200)))

  def test_iter_dataset_docstring_example(self):
    # This test verifies the example provided in the ElasticIterator docstring.
    source_ds = dataset.MapDataset.range(1000)
    shard_opts = sharding.ShardOptions(shard_index=0, shard_count=4)
    elastic_iter = elastic_iterator.ElasticIterator(
        ds=source_ds,
        global_batch_size=128,
        shard_options=shard_opts,
    )
    batch = next(elastic_iter)
    self.assertLen(batch, 32)


if __name__ == "__main__":
  absltest.main()
