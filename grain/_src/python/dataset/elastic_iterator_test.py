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

from absl.testing import parameterized
from grain._src.core import sharding
from grain._src.python.dataset import dataset
from grain._src.python.dataset import elastic_iterator
import grain._src.python.testing.experimental as test_util
import numpy as np
from absl.testing import absltest


class ElasticIteratorTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          global_batch_size=5,
          shard_options=sharding.NoSharding(),
          expected=[np.arange(1, 6), np.arange(6, 11)],
      ),
      dict(
          global_batch_size=2,
          shard_options=sharding.ShardOptions(shard_index=0, shard_count=2),
          expected=[[1], [3], [5], [7], [9]],
      ),
      dict(
          global_batch_size=8,
          shard_options=sharding.ShardOptions(shard_index=2, shard_count=4),
          expected=[[3, 7]],
      ),
  )
  def test_produces_correct_elements(
      self, global_batch_size, shard_options, expected
  ):
    ds = dataset.MapDataset.range(10).map(lambda x: x + 1)
    actual = list(
        elastic_iterator.ElasticIterator(ds, global_batch_size, shard_options)
    )
    np.testing.assert_equal(
        actual, expected, err_msg=f"actual: {actual}, expected: {expected}"
    )

  def test_checkpointing(self):
    ds = dataset.MapDataset.range(10).map(lambda x: x * 2).shuffle(42)
    it = elastic_iterator.ElasticIterator(ds, 5, sharding.NoSharding())
    test_util.assert_equal_output_after_checkpoint(it)

  def test_elastic_downsize(self):
    ds = dataset.MapDataset.range(1024).map(lambda x: x * 2).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))
    # Create iterators over 32 hosts with per-host batch size 2.
    iterators = [
        elastic_iterator.ElasticIterator(
            ds, 64, sharding.ShardOptions(shard_index=i, shard_count=32)
        )
        for i in range(32)
    ]
    # Advance all iterators by 7 steps.
    all_actual_elements = set()
    for it in iterators:
      for _ in range(7):
        element = next(it)
        self.assertEqual(element.shape[0], 2)
        all_actual_elements.update(element.tolist())
    # Get state of any iterator.
    state = iterators[0].get_state()
    # Create new iterators over 16 hosts with per-host batch size 2.
    iterators = [
        elastic_iterator.ElasticIterator(
            ds, 32, sharding.ShardOptions(shard_index=i, shard_count=16)
        )
        for i in range(16)
    ]
    # Restore state of all iterators.
    for it in iterators:
      it.set_state(state)
    # Advance all iterators by the remaining steps.
    for it in iterators:
      for element in it:
        self.assertEqual(element.shape[0], 2)
        all_actual_elements.update(element.tolist())

    # Check that all elements were produced exactly once.
    self.assertSetEqual(all_actual_elements, all_expected_elements)

  def test_elastic_upsize(self):
    ds = dataset.MapDataset.range(2048).map(lambda x: x - 1).shuffle(42)
    all_expected_elements = set(ds)
    self.assertLen(all_expected_elements, len(ds))
    # Create iterators over 8 hosts with per-host batch size 4.
    iterators = [
        elastic_iterator.ElasticIterator(
            ds, 32, sharding.ShardOptions(shard_index=i, shard_count=8)
        )
        for i in range(8)
    ]
    # Advance all iterators by 8 steps.
    all_actual_elements = set()
    for it in iterators:
      for _ in range(8):
        element = next(it)
        self.assertEqual(element.shape[0], 4)
        all_actual_elements.update(element.tolist())
    # Get state of any iterator.
    state = iterators[0].get_state()
    # Create new iterators over 64 hosts with per-host batch size 4.
    iterators = [
        elastic_iterator.ElasticIterator(
            ds, 256, sharding.ShardOptions(shard_index=i, shard_count=64)
        )
        for i in range(64)
    ]
    # Restore state of all iterators.
    for it in iterators:
      it.set_state(state)
    # Advance all iterators by the remaining steps.
    for it in iterators:
      for element in it:
        self.assertEqual(element.shape[0], 4)
        all_actual_elements.update(element.tolist())

    # Check that all elements were produced exactly once.
    self.assertSetEqual(all_actual_elements, all_expected_elements)

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
