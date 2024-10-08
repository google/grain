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

import sys

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import packing
from grain._src.python.dataset.transformations import source
import numpy as np

import tree


_IS_PY310 = sys.version_info >= (3, 10)


def _assert_trees_equal(actual, expected):
  def _check_equivalence(path, actual_val, expected_val):
    np.testing.assert_array_equal(
        actual_val,
        expected_val,
        err_msg=(
            f"Pytrees differ at path {path}.\n\n"
            f"Actual: {actual_val}\n\nExpected: {expected_val}"
        ),
    )

  tree.map_structure_with_path(_check_equivalence, actual, expected)


class SingleBinPackIterDatasetTest(parameterized.TestCase):

  def test_pack_single_feature(self):
    # 5 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7], [8]]
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(ds, length_struct=4)
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
    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
      # Elements are tuples with (inputs, inputs_segment_ids, inputs_positions).
      self.assertLen(actual, 3)
      np.testing.assert_array_equal(actual, expected)

  def test_pack_single_feature_remainder_is_padded(self):
    # 4 elements of variable sequence length.
    input_elements = [[1, 2, 3, 4], [5, 6], [11, 12, 13, 14], [7]]
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(np.asarray)
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(ds, length_struct=4)
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

    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
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
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(ds, length_struct={"inputs": 4})
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

    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
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
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(
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
    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
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
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(
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
    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
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
    ds = source.SourceMapDataset(input_elements)
    ds = ds.map(lambda d: {k: np.asarray(v) for k, v in d.items()})
    ds = ds.to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(
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
    for actual, expected in zip(
        ds_iter, expected_elements, **({"strict": True} if _IS_PY310 else {})
    ):
      np.testing.assert_array_equal(actual[feature], expected[feature])

  def test_checkpointing(self):
    ds = dataset.MapDataset.range(1, 12).map(
        lambda x: {"x": 2**x + np.arange(x)}
    )
    ds = ds.shuffle(seed=3)
    ds = ds.repeat(None).to_iter_dataset()
    ds = packing.SingleBinPackIterDataset(ds, length_struct={"x": 8})
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


def _common_test_body(
    input_elements,
    expected_elements,
    length_struct,
    *,
    num_packing_bins: int,
    shuffle_bins: bool = False,
):
  """Factor out common test operations in a separate function."""
  input_elements = [
      {k: np.asarray(v) for k, v in d.items()} for d in input_elements
  ]
  expected_elements = [
      {k: np.asarray(v) for k, v in d.items()} for d in expected_elements
  ]
  ld = packing.FirstFitPackIterDataset(
      source.SourceMapDataset(input_elements).to_iter_dataset(),
      num_packing_bins=num_packing_bins,
      length_struct=length_struct,
      shuffle_bins=shuffle_bins,
  )
  actual_elements = list(ld)
  np.testing.assert_equal(len(actual_elements), len(expected_elements))

  def _check_equivalence(path, actual_val, expected_val):
    np.testing.assert_array_equal(
        actual_val,
        expected_val,
        err_msg=(
            f"Pytrees differ at path {path}.\n\n"
            f"Actual: {actual_val}\n\nExpected: {expected_val}"
        ),
    )

  for actual, expected in zip(actual_elements, expected_elements):
    tree.map_structure_with_path(_check_equivalence, actual, expected)


class FirstFitPackIterDatasetTest(parameterized.TestCase):
  """Tests for FirstFitPackIterDataset."""

  @parameterized.parameters(
      {"num_packing_bins": 1},
      {"num_packing_bins": 2},
      {"num_packing_bins": 3},
  )
  def test_pack_sequences_length_3(self, num_packing_bins: int):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]

    length_struct = {"inputs": 3, "targets": 3}

    expected_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0],
            "inputs_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0],
        },
        {
            "inputs": [4, 5, 0],
            "targets": [20, 30, 40],
            "inputs_segment_ids": [1, 1, 0],
            "targets_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2],
        },
        {
            "inputs": [6, 0, 0],
            "targets": [50, 60, 0],
            "inputs_segment_ids": [1, 0, 0],
            "targets_segment_ids": [1, 1, 0],
            "inputs_positions": [0, 0, 0],
            "targets_positions": [0, 1, 0],
        },
    ]

    _common_test_body(
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=num_packing_bins,
    )

  @parameterized.parameters(
      {"num_packing_bins": 3},
      {"num_packing_bins": 5},
  )
  def test_pack_sequences_length_shuffle_bins(self, num_packing_bins: int):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]

    length_struct = {"inputs": 3, "targets": 3}

    expected_elements = [
        {
            "inputs": [6, 0, 0],
            "targets": [50, 60, 0],
            "inputs_segment_ids": [1, 0, 0],
            "targets_segment_ids": [1, 1, 0],
            "inputs_positions": [0, 0, 0],
            "targets_positions": [0, 1, 0],
        },
        {
            "inputs": [1, 2, 3],
            "targets": [10, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0],
            "inputs_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0],
        },
        {
            "inputs": [4, 5, 0],
            "targets": [20, 30, 40],
            "inputs_segment_ids": [1, 1, 0],
            "targets_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2],
        },
    ]

    _common_test_body(
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=num_packing_bins,
        shuffle_bins=True,
    )

  def test_pack_sequences_length_4(self):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    length_struct = {"inputs": 4, "targets": 4}

    expected_elements = [
        {
            "inputs": [1, 2, 3, 6],
            "targets": [10, 50, 60, 0],
            "inputs_segment_ids": [1, 1, 1, 2],
            "targets_segment_ids": [1, 2, 2, 0],
            "inputs_positions": [0, 1, 2, 0],
            "targets_positions": [0, 0, 1, 0],
        },
        {
            "inputs": [4, 5, 0, 0],
            "targets": [20, 30, 40, 0],
            "inputs_segment_ids": [1, 1, 0, 0],
            "targets_segment_ids": [1, 1, 1, 0],
            "inputs_positions": [0, 1, 0, 0],
            "targets_positions": [0, 1, 2, 0],
        },
    ]

    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=2
    )

  def test_pack_sequences_length_5(self):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    length_struct = {"inputs": 5, "targets": 5}

    expected_elements = [
        {
            "inputs": [1, 2, 3, 4, 5],
            "targets": [10, 20, 30, 40, 0],
            "inputs_segment_ids": [1, 1, 1, 2, 2],
            "targets_segment_ids": [1, 2, 2, 2, 0],
            "inputs_positions": [0, 1, 2, 0, 1],
            "targets_positions": [0, 0, 1, 2, 0],
        },
        {
            "inputs": [6, 0, 0, 0, 0],
            "targets": [50, 60, 0, 0, 0],
            "inputs_segment_ids": [1, 0, 0, 0, 0],
            "targets_segment_ids": [1, 1, 0, 0, 0],
            "inputs_positions": [0, 0, 0, 0, 0],
            "targets_positions": [0, 1, 0, 0, 0],
        },
    ]

    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=2
    )

  def test_pack_sequences_length_6(self):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    length_struct = {"inputs": 6, "targets": 6}

    expected_elements = [{
        "inputs": [1, 2, 3, 4, 5, 6],
        "targets": [10, 20, 30, 40, 50, 60],
        "inputs_segment_ids": [1, 1, 1, 2, 2, 3],
        "targets_segment_ids": [1, 2, 2, 2, 3, 3],
        "inputs_positions": [0, 1, 2, 0, 1, 0],
        "targets_positions": [0, 0, 1, 2, 0, 1],
    }]

    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=2
    )

  def test_pack_sequences_length_7(self):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    length_struct = {"inputs": 7, "targets": 7}

    expected_elements = [{
        "inputs": [1, 2, 3, 4, 5, 6, 0],
        "targets": [10, 20, 30, 40, 50, 60, 0],
        "inputs_segment_ids": [1, 1, 1, 2, 2, 3, 0],
        "targets_segment_ids": [1, 2, 2, 2, 3, 3, 0],
        "inputs_positions": [0, 1, 2, 0, 1, 0, 0],
        "targets_positions": [0, 0, 1, 2, 0, 1, 0],
    }]

    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=1
    )

  def test_pack_sequences_different_lengths(self):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    length_struct = {"inputs": 3, "targets": 4}

    expected_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10, 0, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0, 0],
            "inputs_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0, 0],
        },
        {
            "inputs": [4, 5, 0],
            "targets": [20, 30, 40, 0],
            "inputs_segment_ids": [1, 1, 0],
            "targets_segment_ids": [1, 1, 1, 0],
            "inputs_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2, 0],
        },
        {
            "inputs": [6, 0, 0],
            "targets": [50, 60, 0, 0],
            "inputs_segment_ids": [1, 0, 0],
            "targets_segment_ids": [1, 1, 0, 0],
            "inputs_positions": [0, 0, 0],
            "targets_positions": [0, 1, 0, 0],
        },
    ]
    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=3
    )

  def test_pack_sequences_two_dimensional_features(self):
    input_elements = [
        {
            "input_tokens": [1, 2, 3],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            "targets": [10],
        },
        {
            "input_tokens": [4, 5],
            "input_vectors": [[3, 4, 5], [4, 5, 6]],
            "targets": [20, 30, 40],
        },
        {
            "input_tokens": [6],
            "input_vectors": [[5, 6, 7]],
            "targets": [50, 60],
        },
    ]

    length_struct = {"input_tokens": 5, "input_vectors": 3, "targets": 5}

    expected_elements = [
        {
            "input_tokens": [1, 2, 3, 0, 0],
            "input_vectors": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            "targets": [10, 0, 0, 0, 0],
            "input_tokens_segment_ids": [1, 1, 1, 0, 0],
            "input_vectors_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0, 0, 0],
            "input_tokens_positions": [0, 1, 2, 0, 0],
            "input_vectors_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0, 0, 0],
        },
        {
            "input_tokens": [4, 5, 6, 0, 0],
            "input_vectors": [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
            "targets": [20, 30, 40, 50, 60],
            "input_tokens_segment_ids": [1, 1, 2, 0, 0],
            "input_vectors_segment_ids": [1, 1, 2],
            "targets_segment_ids": [1, 1, 1, 2, 2],
            "input_tokens_positions": [0, 1, 0, 0, 0],
            "input_vectors_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2, 0, 1],
        },
    ]

    _common_test_body(
        input_elements, expected_elements, length_struct, num_packing_bins=2
    )

  @parameterized.parameters(
      {"restore_at_step": 0},
      {"restore_at_step": 1},
      {"restore_at_step": 2},
      {"restore_at_step": 3},
  )
  def test_checkpointing(self, restore_at_step: int):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "inputs": [6],
            "targets": [50, 60],
        },
    ]
    input_elements = [
        {k: np.asarray(v) for k, v in d.items()} for d in input_elements
    ]
    length_struct = {"inputs": 3, "targets": 3}
    ld = packing.FirstFitPackIterDataset(
        source.SourceMapDataset(input_elements).to_iter_dataset(),
        num_packing_bins=2,
        length_struct=length_struct,
        shuffle_bins=True,
    )
    # There will be 3 packed sequences as output.
    data_iter = ld.__iter__()
    actual_elements = []
    for step in range(5):
      state = data_iter.get_state()
      if restore_at_step == step:
        data_iter.set_state(state)
      try:
        next_element = next(data_iter)
      except StopIteration:
        continue
      actual_elements.append(next_element)

    expected_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0],
            "inputs_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0],
        },
        {
            "inputs": [4, 5, 0],
            "targets": [20, 30, 40],
            "inputs_segment_ids": [1, 1, 0],
            "targets_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2],
        },
        {
            "inputs": [6, 0, 0],
            "targets": [50, 60, 0],
            "inputs_segment_ids": [1, 0, 0],
            "targets_segment_ids": [1, 1, 0],
            "inputs_positions": [0, 0, 0],
            "targets_positions": [0, 1, 0],
        },
    ]
    expected_elements = [
        {k: np.asarray(v) for k, v in d.items()} for d in expected_elements
    ]

    np.testing.assert_equal(len(actual_elements), len(expected_elements))
    for actual, expected in zip(actual_elements, expected_elements):
      _assert_trees_equal(actual, expected)

  @parameterized.product(
      shuffle_bins=[True, False],
  )
  def test_deterministic_restore(self, shuffle_bins: bool):
    # Tests whether the dataset is deterministic after checkpointing+restore.
    steps_to_skip = 10
    examples_to_compare = 3

    rng = np.random.default_rng(42)
    elements = [
        dict(row=rng.integers(0, 10, size=rng.integers(5, 30)))
        for _ in range(100)
    ]
    ld = packing.FirstFitPackIterDataset(
        source.SourceMapDataset(elements).repeat().to_iter_dataset(),
        num_packing_bins=4,
        length_struct=dict(row=100),
        shuffle_bins=shuffle_bins,
    )

    it = ld.__iter__()

    for _ in range(steps_to_skip):
      _ = next(it)
    state = it.get_state()

    first_elements = [next(it) for _ in range(examples_to_compare)]

    it = ld.__iter__()
    it.set_state(state)

    second_elements = [next(it) for _ in range(examples_to_compare)]

    _assert_trees_equal(first_elements, second_elements)

  @parameterized.product(
      mark_as_meta_feature=[True, False],
  )
  def test_nested_feature(self, mark_as_meta_feature: bool):
    # Nested features must be marked as meta features since we don't support
    # extracting their segment ids and positions.
    rng = np.random.default_rng(42)
    elements = [
        dict(
            row=rng.integers(0, 10, size=rng.integers(5, 30)),
            nested_feature=dict(
                inner_value=0,
            ),
        )
        for _ in range(100)
    ]
    ld = packing.FirstFitPackIterDataset(
        source.SourceMapDataset(elements).repeat().to_iter_dataset(),
        num_packing_bins=4,
        length_struct=dict(row=100, nested_feature=dict(inner_value=100)),
        meta_features=["nested_feature"] if mark_as_meta_feature else [],
    )
    if mark_as_meta_feature:
      # No error when nested feature is marked as a meta feature.
      _ = next(iter(ld))
    else:
      with self.assertRaisesRegex(
          ValueError,
          "Failed to extract segment ids for 'nested_feature', which has type"
          " <class 'dict'> rather than np.ndarray. Perhaps it should be marked"
          " as a meta feature?",
      ):
        _ = next(iter(ld))


if __name__ == "__main__":
  absltest.main()
