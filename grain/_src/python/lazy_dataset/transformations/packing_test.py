# Copyright 2023 Google LLC
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
"""Tests for batch transformation."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.lazy_dataset import data_sources
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import packing
# pylint: disable=unused-import
import grain._src.python.lazy_dataset.transformations.map
import grain._src.python.lazy_dataset.transformations.repeat
import grain._src.python.lazy_dataset.transformations.shuffle
# pylint: enable=unused-import
import numpy as np


class SingleBinPackLazyIterDatasetTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7], [8]]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(ds, length_struct=4)
    ds_iter = iter(ds)

    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second element is in buffer and we yield the third element first
        # because it's already fully packed on 'inputs'.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second, fourth and five element packed together.
        ([5, 6, 7, 8], [1, 1, 2, 3], [0, 1, 0, 0]),
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      self.assertLen(actual, 3)
      np.testing.assert_array_equal(actual, expected)

  def test_pack_single_feature_remainder_is_padded(self):
    # 4 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(ds, length_struct=4)
    ds_iter = iter(ds)

    # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
    expected_elements = [
        # First element was already fully packed on 'inputs'.
        ([1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second element is in buffer and we yield the third element first
        # because it's already fully packed on 'inputs'.
        ([11, 12, 13, 14], [1, 1, 1, 1], [0, 1, 2, 3]),
        # Second and fourth element packed together (plus padding).
        ([5, 6, 7, 0], [1, 1, 2, 0], [0, 1, 0, 0]),
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      self.assertLen(actual, 3)
      np.testing.assert_array_equal(actual, expected)

  # Same as above but elements are dictionaries.
  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
  )
  def test_pack_single_feature_in_dict(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
        },
        {
            "inputs": [5, 6],
        },
        {
            "inputs": [11, 12, 13, 14],
        },
        {
            "inputs": [7],
        },
        {
            "inputs": [8],
        },
    ]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(ds, length_struct={"inputs": 4})
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second element is in buffer and we yield the third element first
        # because it's already fully packed on 'inputs'.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
        },
        # Second, fourth and fifth element packed together.
        {
            "inputs": [5, 6, 7, 8],
            "inputs_segment_ids": [1, 1, 2, 3],
            "inputs_positions": [0, 1, 0, 0],
        },
    ]

    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_same_sequences_length(self, feature: str):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(
        ds, length_struct={"inputs": 4, "targets": 4}
    )
    ds_iter = iter(ds)

    expected_elements = [
        # First element was already fully packed on 'inputs'.
        {
            "inputs": [1, 2, 3, 4],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets": [10, 20, 0, 0],
            "targets_segment_ids": [1, 1, 0, 0],
            "targets_positions": [0, 1, 0, 0],
        },
        # Second element is in buffer and we yield the third element first
        # because it's already fully packed on 'inputs'.
        {
            "inputs": [11, 12, 13, 14],
            "inputs_segment_ids": [1, 1, 1, 1],
            "inputs_positions": [0, 1, 2, 3],
            "targets": [31, 41, 51, 0],
            "targets_segment_ids": [1, 1, 1, 0],
            "targets_positions": [0, 1, 2, 0],
        },
        # Second and fourth element packed together.
        {
            "inputs": [5, 6, 7, 0],
            "inputs_segment_ids": [1, 1, 2, 0],
            "inputs_positions": [0, 1, 0, 0],
            "targets": [30, 40, 50, 60],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      # Compare keys.
      self.assertSequenceEqual(sorted(actual), sorted(expected))
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "inputs",
      "inputs_segment_ids",
      "inputs_positions",
      "targets",
      "targets_segment_ids",
      "targets_positions",
  )
  def test_pack_multiple_features_different_sequences_length(
      self, feature: str
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3, 4],
            "targets": [10, 20],
        },
        {
            "inputs": [5, 6],
            "targets": [30, 40, 50],
        },
        {
            "inputs": [11, 12, 13, 14],
            "targets": [31, 41, 51],
        },
        {
            "inputs": [7],
            "targets": [60],
        },
    ]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(
        ds, length_struct={"inputs": 6, "targets": 4}
    )
    ds_iter = iter(ds)

    expected_elements = [
        {
            "inputs": [1, 2, 3, 4, 5, 6],
            "inputs_segment_ids": [1, 1, 1, 1, 2, 2],
            "inputs_positions": [0, 1, 2, 3, 0, 1],
            "targets": [10, 20, 30, 40],  # 50 gets dropped.
            "targets_segment_ids": [1, 1, 2, 2],
            "targets_positions": [0, 1, 0, 1],
        },
        {
            "inputs": [11, 12, 13, 14, 7, 0],
            "inputs_segment_ids": [1, 1, 1, 1, 2, 0],
            "inputs_positions": [0, 1, 2, 3, 0, 0],
            "targets": [31, 41, 51, 60],
            "targets_segment_ids": [1, 1, 1, 2],
            "targets_positions": [0, 1, 2, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  @parameterized.parameters(
      "input_tokens",
      "input_tokens_segment_ids",
      "input_tokens_positions",
      "input_vectors",
      "input_vectors_segment_ids",
      "input_vectors_positions",
  )
  def test_pack_two_dimensional_features(self, feature: str):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
        },
    ]
    ds = data_sources.SourceLazyMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(
        ds, length_struct={"input_tokens": 3, "input_vectors": 3}
    )
    ds_iter = iter(ds)

    expected_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_tokens_segment_ids": [1, 1, 1],
            "input_tokens_positions": [0, 1, 2],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            "input_vectors_segment_ids": [1, 1, 1],
            "input_vectors_positions": [0, 1, 2],
        },
        {
            "input_tokens": [4, 5, 6],
            "input_tokens_segment_ids": [1, 1, 2],
            "input_tokens_positions": [0, 1, 0],
            "input_vectors": [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
            "input_vectors_segment_ids": [1, 1, 2],
            "input_vectors_positions": [0, 1, 0],
        },
    ]
    for actual, expected in zip(ds_iter, expected_elements, strict=True):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_checkpointing(self):
    ds = lazy_dataset.RangeLazyMapDataset(1, 12).map(
        lambda x: {"x": 2**x + np.arange(x)}
    )
    ds = ds.shuffle(seed=3)
    ds = ds.repeat(None).to_iter_dataset()
    ds = packing.SingleBinPackLazyIterDataset(ds, length_struct={"x": 8})
    ds_iter = iter(ds)

    max_steps = 10
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 1, 5, 8]:
      ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        for k, v in value.items():
          np.testing.assert_array_equal(v, values_without_interruption[i][k])


if __name__ == "__main__":
  absltest.main()
