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
"""Tests for zip transformation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import source
import grain._src.python.dataset.transformations.zip as zip_ds
import grain._src.python.testing.experimental as test_util
import numpy as np


class ZipMapDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_list = [
        dataset.MapDataset.range(0, 20),
        dataset.MapDataset.range(1, 21),
        dataset.MapDataset.range(2, 22),
    ]

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in (
          list(itertools.combinations(range(3), 3))
          + list(itertools.combinations(range(3), 2))
          + list(itertools.combinations(range(3), 1))
      )
  )
  def test_len(self, ds_idx_list):
    self.assertLen(
        zip_ds.ZipMapDataset(parents=[self.ds_list[i] for i in ds_idx_list]),
        20,
    )

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in (
          list(itertools.combinations(range(3), 3))
          + list(itertools.combinations(range(3), 2))
          + list(itertools.combinations(range(3), 1))
      )
  )
  def test_getitem(self, ds_idx_list):
    ds = zip_ds.ZipMapDataset(parents=[self.ds_list[i] for i in ds_idx_list])
    for i in range(20):
      self.assertEqual(ds[i], tuple(i + ds_idx for ds_idx in ds_idx_list))

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in (
          list(itertools.combinations(range(3), 3))
          + list(itertools.combinations(range(3), 2))
          + list(itertools.combinations(range(3), 1))
      )
  )
  def test_getitems(self, ds_idx_list):
    ds = zip_ds.ZipMapDataset(parents=[self.ds_list[i] for i in ds_idx_list])
    indices = [0, 5, 19]
    expected_elements = [ds[i] for i in indices]
    actual_elements = ds._getitems(indices)
    self.assertEqual(expected_elements, actual_elements)

  def test_example_docstring(self):
    inputs_ds = dataset.MapDataset.source([10, 20, 30])
    labels_ds = dataset.MapDataset.source([40, 50, 60])
    zipped_ds = zip_ds.ZipMapDataset([inputs_ds, labels_ds])
    self.assertEqual(zipped_ds[0], (10, 40))
    self.assertEqual(zipped_ds[1], (20, 50))

  def test_element_spec(self):
    ds = zip_ds.ZipMapDataset(
        parents=[dataset.MapDataset.range(2), dataset.MapDataset.range(2)]
    )
    specs = dataset.get_element_spec(ds)
    for spec in specs:
      self.assertEqual(spec.dtype, np.int64)
      self.assertEqual(spec.shape, ())

  def test_dict(self):
    source1 = source.SourceMapDataset(
        [{"a": [1, 2]}, {"b": [3]}, {"c": [4, 5, 6]}]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertLen(list(source1), 3)
    source2 = source.SourceMapDataset(
        [{"d": [7]}, {"e": [8, 9]}, {"f": [10, 11, 12]}]  # pyrefly: ignore[bad-argument-type]
    )
    self.assertLen(list(source2), 3)
    ds = zip_ds.ZipMapDataset(parents=[source1, source2])
    out = list(ds)
    self.assertLen(out, 3)
    self.assertEqual(
        out,
        [
            ({"a": [1, 2]}, {"d": [7]}),
            ({"b": [3]}, {"e": [8, 9]}),
            ({"c": [4, 5, 6]}, {"f": [10, 11, 12]}),
        ],
    )

  def test_empty_parents(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "At least one parent must be provided."
    ):
      zip_ds.ZipMapDataset(parents=[])

  def test_mismatched_lengths(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "All parents must have the same length."
    ):
      zip_ds.ZipMapDataset(
          parents=[
              dataset.MapDataset.range(5),
              dataset.MapDataset.range(10),
          ]
      )

  def test_slice(self):
    ds = zip_ds.ZipMapDataset(
        parents=[
            dataset.MapDataset.range(0, 10),
            dataset.MapDataset.range(10, 20),
        ]
    )
    sliced_ds = ds[1:7:2]
    self.assertEqual(list(sliced_ds), [(1, 11), (3, 13), (5, 15)])
    self.assertLen(sliced_ds, 3)

  def test_str(self):
    ds = zip_ds.ZipMapDataset(
        parents=[
            dataset.MapDataset.range(5),
            dataset.MapDataset.range(5),
        ]
    )
    self.assertIn("ZipMapDataset", str(ds))


class ZipIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds_list = [
        dataset.MapDataset.range(0, 20),
        dataset.MapDataset.range(1, 21),
        dataset.MapDataset.range(2, 22),
    ]

  @parameterized.parameters(
      {"ds_idx_list": x}
      for x in (
          list(itertools.combinations(range(3), 3))
          + list(itertools.combinations(range(3), 2))
          + list(itertools.combinations(range(3), 1))
      )
  )
  def test_iter(self, ds_idx_list):
    ds = zip_ds.ZipIterDataset(
        parents=[self.ds_list[i].to_iter_dataset() for i in ds_idx_list]
    )
    out = list(ds)
    for i in range(20):
      self.assertEqual(out[i], tuple(i + ds_idx for ds_idx in ds_idx_list))

  def test_strict_zip_shorter(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 2 is shorter than argument 1"
    ):
      list(ds)

  def test_strict_zip_shorter_many(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(3).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 3 is shorter than arguments 1-2"
    ):
      list(ds)

  def test_strict_zip_longer_many(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(3).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 4 is longer than arguments 1-3"
    ):
      list(ds)

  def test_non_strict_zip(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(1, 4).to_iter_dataset(),
        ],
        strict=False,
    )
    actual = list(ds)
    expected = [(0, 1), (1, 2)]
    self.assertEqual(actual, expected)

  def test_checkpointing(self):
    ds = zip_ds.ZipIterDataset(
        parents=[p.to_iter_dataset() for p in self.ds_list]
    )
    test_util.assert_equal_output_after_checkpoint(ds)

  def test_element_spec(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(2).to_iter_dataset(),
        ]
    )
    specs = dataset.get_element_spec(ds)
    for spec in specs:
      self.assertEqual(spec.dtype, np.int64)
      self.assertEqual(spec.shape, ())

  def test_set_slice(self):
    ds1 = [
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset() for i in range(4)
    ]
    interleave_ds1 = interleave.InterleaveIterDataset(ds1, cycle_length=2)

    ds2 = [
        dataset.MapDataset.range(i + 10, 20, 4).to_iter_dataset()
        for i in range(4)
    ]
    interleave_ds2 = interleave.InterleaveIterDataset(ds2, cycle_length=2)

    zipped_ds = zip_ds.ZipIterDataset([interleave_ds1, interleave_ds2])
    zipped_ds.set_slice(slice(0, None, 2))

    actual = list(zipped_ds)
    expected = [(0, 10), (2, 12), (4, 14), (6, 16), (8, 18)]
    self.assertEqual(actual, expected)

  def test_get_set_shard_states(self):
    ds1 = [
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset() for i in range(4)
    ]
    interleave_ds1 = interleave.InterleaveIterDataset(ds1, cycle_length=2)

    ds2 = [
        dataset.MapDataset.range(i + 10, 20, 4).to_iter_dataset()
        for i in range(4)
    ]
    interleave_ds2 = interleave.InterleaveIterDataset(ds2, cycle_length=2)

    zipped_ds = zip_ds.ZipIterDataset([interleave_ds1, interleave_ds2])
    zipped_ds.set_slice(slice(0, None, 2))

    it = zipped_ds.__iter__()
    self.assertEqual(next(it), (0, 10))
    self.assertEqual(next(it), (2, 12))

    shard_states = it.get_shard_states()  # pytype: disable=attribute-error
    self.assertLen(shard_states, 2)

    it2 = zipped_ds.__iter__()
    it2.set_shard_states(shard_states)  # pytype: disable=attribute-error
    self.assertEqual(next(it2), (4, 14))

  def test_get_set_shard_states_nested(self):
    class WrapperIterDataset(dataset.IterDataset):

      def __init__(self, parent):
        self._parents = [parent]

      def __iter__(self):
        return WrapperDatasetIterator(self._parents[0].__iter__())  # pyrefly: ignore[missing-attribute]

    class WrapperDatasetIterator(dataset.DatasetIterator):

      def __init__(self, parent_iter):
        self._parents = [parent_iter]  # pyrefly: ignore[bad-assignment]
        self._ctx = parent_iter._ctx

      def __next__(self):
        return next(self._parents[0])

      def get_state(self):
        return self._parents[0].get_state()

      def set_state(self, state):
        self._parents[0].set_state(state)

    ds1 = [
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset() for i in range(4)
    ]
    interleave_ds1 = interleave.InterleaveIterDataset(ds1, cycle_length=2)

    ds2 = [
        dataset.MapDataset.range(i + 10, 20, 4).to_iter_dataset()
        for i in range(4)
    ]
    interleave_ds2 = interleave.InterleaveIterDataset(ds2, cycle_length=2)

    wrapped_ds1 = WrapperIterDataset(interleave_ds1)
    wrapped_ds2 = WrapperIterDataset(interleave_ds2)

    zipped_ds = zip_ds.ZipIterDataset([wrapped_ds1, wrapped_ds2])
    zipped_ds.set_slice(slice(0, None, 2))

    it = zipped_ds.__iter__()
    self.assertEqual(next(it), (0, 10))
    self.assertEqual(next(it), (2, 12))

    shard_states = it.get_shard_states()  # pytype: disable=attribute-error
    self.assertLen(shard_states, 2)

    it2 = zipped_ds.__iter__()
    it2.set_shard_states(shard_states)  # pytype: disable=attribute-error
    self.assertEqual(next(it2), (4, 14))

  def test_empty_parents(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "At least one parent must be provided."
    ):
      zip_ds.ZipIterDataset(parents=[])

  def test_strict_zip_longer_two_parents(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(2).to_iter_dataset(),
            dataset.MapDataset.range(3).to_iter_dataset(),
        ],
        strict=True,
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "ZipIterDataset argument 2 is longer than argument 1"
    ):
      list(ds)

  def test_get_shard_states_unsupported_parent(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(10).to_iter_dataset(),
            dataset.MapDataset.range(10).to_iter_dataset(),
        ]
    )
    it = iter(ds)
    with self.assertRaisesRegex(
        ValueError, "does not support elastic resizing"
    ):
      it.get_shard_states()  # pytype: disable=attribute-error

  def test_get_shard_states_mismatched_shard_counts(self):
    class DummyElasticDataset(dataset.IterDataset):

      def __init__(self, num_shards):
        super().__init__()
        self._num_shards = num_shards

      def __iter__(self):
        return DummyElasticIterator(self._num_shards)

    class DummyElasticIterator(dataset.DatasetIterator):

      def __init__(self, num_shards):
        super().__init__()
        self._num_shards = num_shards

      def __next__(self):
        raise StopIteration

      def get_state(self):
        return {}

      def set_state(self, state):
        pass

      def get_shard_states(self):
        return [{"state": i} for i in range(self._num_shards)]

      def set_shard_states(self, shard_states):
        pass

    ds = zip_ds.ZipIterDataset(
        parents=[DummyElasticDataset(2), DummyElasticDataset(3)]
    )
    it = iter(ds)
    with self.assertRaisesWithLiteralMatch(
        ValueError, "All parents must have the same number of shards."
    ):
      it.get_shard_states()  # pytype: disable=attribute-error

  def test_set_shard_states_empty(self):
    ds1 = [
        dataset.MapDataset.range(i, 10, 4).to_iter_dataset() for i in range(4)
    ]
    interleave_ds1 = interleave.InterleaveIterDataset(ds1, cycle_length=2)
    zipped_ds = zip_ds.ZipIterDataset([interleave_ds1, interleave_ds1])
    it = iter(zipped_ds)
    it.set_shard_states([])  # pytype: disable=attribute-error

  def test_str(self):
    ds = zip_ds.ZipIterDataset(
        parents=[
            dataset.MapDataset.range(5).to_iter_dataset(),
        ]
    )
    self.assertIn("ZipIterDataset", str(ds))
    it = iter(ds)
    self.assertIn("ZipDatasetIterator", str(it))


if __name__ == "__main__":
  absltest.main()
