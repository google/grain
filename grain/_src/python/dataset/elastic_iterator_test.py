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

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import sharding
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset import elastic_iterator
import grain._src.python.testing.experimental as test_util
import numpy as np


class ElasticIteratorTest(parameterized.TestCase):

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
      elastic_iterator.ElasticIterator(ds, 5, sharding.NoSharding())


if __name__ == "__main__":
  absltest.main()
