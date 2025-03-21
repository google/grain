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
"""Unit tests for ConcatThenSplitIterDataset."""

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import exceptions
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import packing_concat_then_split
from grain._src.python.dataset.transformations import source
from grain._src.python.testing import experimental
import numpy as np


BOSHandling = packing_concat_then_split.BOSHandling


# Pretty print elements for debugging.
def _print_elements(elements: list[dict[str, np.ndarray]]):
  s = "\n"
  for element in elements:
    elements_str = "\n".join(
        [f'   "{k}": {v.tolist()}),' for k, v in element.items()]
    )
    s += f"{{\n{elements_str}\n}},\n"
  return s


class ConcatThenSplitIterDatasetTest(parameterized.TestCase):

  # observations will be [
  #   [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5],
  #   [6, 6, 6, 6, 6, 6], [1], [2, 2], [3, 3, 3], ...
  # ].
  def dummy_iter_dataset(self, *, num_observations: int) -> dataset.IterDataset:
    return (
        source.RangeMapDataset(1, 7)
        .repeat()
        .map_with_index(
            lambda index, value: {
                "observation": np.repeat(value, value),
                "index": index + 1,
            }
        )[:num_observations]
        .to_iter_dataset()
    )

  def test_meta_features_not_restricting_when_splitting_full_length_features(
      self,
  ):
    # Pack 9 elements.
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=9),
        length_struct={"observation": 6, "index": 6},
        meta_features={"index"},
        split_full_length_features=True,
    )
    actual_elements = list(ds)
    self.assert_equal_elements(
        actual_elements,
        [
            {
                "observation": np.asarray([1, 2, 2, 3, 3, 3]),
                "observation_segment_ids": np.asarray([1, 2, 2, 3, 3, 3]),
                "observation_positions": np.asarray([0, 0, 1, 0, 1, 2]),
                "index": np.asarray([1, 2, 3, 0, 0, 0]),
            },
            # 5 gets split between the two packed elements.
            {
                "observation": np.asarray([4, 4, 4, 4, 5, 5]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 3, 0, 1]),
                "index": np.asarray([4, 5, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([5, 5, 5, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 0, 1, 2]),
                "index": np.asarray([5, 6, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([6, 6, 6, 1, 2, 2]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 3, 3]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 1]),
                "index": np.asarray([6, 7, 8, 0, 0, 0]),
            },
            # Reached end.
            {
                "observation": np.asarray([3, 3, 3, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 1, 1, 0, 0, 0]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 0]),
                "index": np.asarray([9, 0, 0, 0, 0, 0]),
            },
        ],
    )

  def test_meta_features_not_restricting(self):
    # Pack 9 elements.
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=9),
        length_struct={"observation": 6, "index": 6},
        meta_features={"index"},
        split_full_length_features=False,
    )
    actual_elements = list(ds)
    self.assert_equal_elements(
        actual_elements,
        [
            {
                "observation": np.asarray([1, 2, 2, 3, 3, 3]),
                "observation_segment_ids": np.asarray([1, 2, 2, 3, 3, 3]),
                "observation_positions": np.asarray([0, 0, 1, 0, 1, 2]),
                "index": np.asarray([1, 2, 3, 0, 0, 0]),
            },
            {
                "observation": np.asarray([4, 4, 4, 4, 5, 5]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 3, 0, 1]),
                "index": np.asarray([4, 5, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([5, 5, 5, 1, 2, 2]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 3, 3]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 1]),
                "index": np.asarray([5, 7, 8, 0, 0, 0]),
            },
            # Fully packed example comes without being split.
            {
                "observation": np.asarray([6, 6, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 1]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 5]),
                "index": np.asarray([6, 0, 0, 0, 0, 0]),
            },
            # Reached end.
            {
                "observation": np.asarray([3, 3, 3, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 1, 1, 0, 0, 0]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 0]),
                "index": np.asarray([9, 0, 0, 0, 0, 0]),
            },
        ],
    )

  def test_meta_features_restricting(self):
    # Pack 9 elements.
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=9),
        length_struct={"observation": 6, "index": 2},
        meta_features={"index"},
        split_full_length_features=False,
    )
    actual_elements = list(ds)
    self.assert_equal_elements(
        actual_elements,
        [
            {
                "observation": np.asarray([1, 2, 2, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 2, 2, 0, 0, 0]),
                "observation_positions": np.asarray([0, 0, 1, 0, 0, 0]),
                "index": np.asarray([1, 2]),
            },
            {
                "observation": np.asarray([3, 3, 3, 4, 4, 4]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 0, 1, 2]),
                "index": np.asarray([3, 4]),
            },
            {
                "observation": np.asarray([4, 5, 5, 5, 5, 5]),
                "observation_segment_ids": np.asarray([1, 2, 2, 2, 2, 2]),
                "observation_positions": np.asarray([0, 0, 1, 2, 3, 4]),
                "index": np.asarray([4, 5]),
            },
            {
                "observation": np.asarray([6, 6, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 1]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 5]),
                "index": np.asarray([6, 0]),
            },
            {
                "observation": np.asarray([1, 2, 2, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 2, 2, 0, 0, 0]),
                "observation_positions": np.asarray([0, 0, 1, 0, 0, 0]),
                "index": np.asarray([7, 8]),
            },
            {
                "observation": np.asarray([3, 3, 3, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 1, 1, 0, 0, 0]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 0]),
                "index": np.asarray([9, 0]),
            },
        ],
    )

  def test_replace_first_token_with_bos(self):
    # Pack 9 elements.
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=9),
        length_struct={"observation": 6, "index": 6},
        meta_features={"index"},
        split_full_length_features=False,
        bos_handling=BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS,
        bos_token_id=1000,
        bos_features={"observation"},
    )
    actual_elements = list(ds)
    self.assert_equal_elements(
        actual_elements,
        [
            {
                "observation": np.asarray([1000, 1000, 2, 1000, 3, 3]),
                "observation_segment_ids": np.asarray([1, 2, 2, 3, 3, 3]),
                "observation_positions": np.asarray([0, 0, 1, 0, 1, 2]),
                "index": np.asarray([1, 2, 3, 0, 0, 0]),
            },
            {
                "observation": np.asarray([1000, 4, 4, 4, 1000, 5]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 3, 0, 1]),
                "index": np.asarray([4, 5, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([1000, 5, 5, 1000, 1000, 2]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 3, 3]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 1]),
                "index": np.asarray([5, 7, 8, 0, 0, 0]),
            },
            # Fully packed example comes without being split.
            {
                "observation": np.asarray([1000, 6, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 1]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 5]),
                "index": np.asarray([6, 0, 0, 0, 0, 0]),
            },
            # Reached end.
            {
                "observation": np.asarray([1000, 3, 3, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 1, 1, 0, 0, 0]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 0]),
                "index": np.asarray([9, 0, 0, 0, 0, 0]),
            },
        ],
    )

  def test_additional_sequence_length(self):
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=7),
        length_struct={
            "observation": 6,
            "index": 6,
            "does_not_exist": 6,
        },
        meta_features={"index"},
    )
    with self.assertRaisesRegex(ValueError, "Parent element has structure"):
      list(ds)

  def test_missingsequence_length(self):
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        self.dummy_iter_dataset(num_observations=7),
        length_struct={"index": 6},
        meta_features={"index"},
    )
    with self.assertRaisesRegex(ValueError, "Parent element has structure"):
      list(ds)

  @parameterized.product(
      checkpoint_steps=[{}, {2}, {0, 3}, {0, 1, 2, 3, 4, 5}],
      split_full_length_features=[True, False],
  )
  def test_checkpointing(
      self, checkpoint_steps: set[int], split_full_length_features: bool
  ):
    def _create_iter(state: dict[str, Any] | None):
      ds = packing_concat_then_split.ConcatThenSplitIterDataset(
          self.dummy_iter_dataset(num_observations=12),
          length_struct={"observation": 8, "index": 6},
          meta_features={"index"},
          split_full_length_features=split_full_length_features,
      )
      ds_iter = ds.__iter__()
      if state is not None:
        ds_iter.set_state(state)
      return ds_iter

    ds_iter = _create_iter(None)
    actual_elements = []
    for step in range(6):
      if step in checkpoint_steps:
        state = ds_iter.get_state()
        ds_iter = _create_iter(state)
      actual_elements.append(next(ds_iter))

    self.assert_equal_elements(
        actual_elements,
        [
            {
                "observation": np.asarray([1, 2, 2, 3, 3, 3, 4, 4]),
                "observation_segment_ids": np.asarray([1, 2, 2, 3, 3, 3, 4, 4]),
                "observation_positions": np.asarray([0, 0, 1, 0, 1, 2, 0, 1]),
                "index": np.asarray([1, 2, 3, 4, 0, 0]),
            },
            {
                "observation": np.asarray([4, 4, 5, 5, 5, 5, 5, 6]),
                "observation_segment_ids": np.asarray([1, 1, 2, 2, 2, 2, 2, 3]),
                "observation_positions": np.asarray([0, 1, 0, 1, 2, 3, 4, 0]),
                "index": np.asarray([4, 5, 6, 0, 0, 0]),
            },
            {
                "observation": np.asarray([6, 6, 6, 6, 6, 1, 2, 2]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 2, 3, 3]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 0, 0, 1]),
                "index": np.asarray([6, 7, 8, 0, 0, 0]),
            },
            {
                "observation": np.asarray([3, 3, 3, 4, 4, 4, 4, 5]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 2, 2, 2, 3]),
                "observation_positions": np.asarray([0, 1, 2, 0, 1, 2, 3, 0]),
                "index": np.asarray([9, 10, 11, 0, 0, 0]),
            },
            {
                "observation": np.asarray([5, 5, 5, 5, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 2, 2, 2, 2]),
                "observation_positions": np.asarray([0, 1, 2, 3, 0, 1, 2, 3]),
                "index": np.asarray([11, 12, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([6, 6, 0, 0, 0, 0, 0, 0]),
                "observation_segment_ids": np.asarray([1, 1, 0, 0, 0, 0, 0, 0]),
                "observation_positions": np.asarray([0, 1, 0, 0, 0, 0, 0, 0]),
                "index": np.asarray([12, 0, 0, 0, 0, 0]),
            },
        ],
    )

  @parameterized.product(
      num_observations=list(range(12)),
      recreate_iter=[True, False],
      split_full_length_features=[True, False],
  )
  def test_checkpointing_after_stop_iteration(
      self,
      num_observations: int,
      recreate_iter: bool,
      split_full_length_features: bool,
  ):
    def _create_iter(state: dict[str, Any] | None):
      ds = packing_concat_then_split.ConcatThenSplitIterDataset(
          self.dummy_iter_dataset(num_observations=num_observations),
          length_struct={"observation": 8, "index": 6},
          meta_features={"index"},
          split_full_length_features=split_full_length_features,
      )
      ds_iter = ds.__iter__()
      if state is not None:
        ds_iter.set_state(state)
      return ds_iter

    ds_iter = _create_iter(None)
    while True:
      try:
        next(ds_iter)
      except StopIteration:
        break

    # Check that we can still restore the checkpoint.
    final_state = ds_iter.get_state()
    if recreate_iter:
      ds_iter = _create_iter(final_state)
    else:
      ds_iter.set_state(final_state)

    with self.assertRaises(StopIteration):
      next(ds_iter)
    # State is still the same.
    self.assertEqual(ds_iter.get_state(), final_state)

  @parameterized.product(
      num_observations=list(range(1, 12)),
      split_full_length_features=[True, False],
  )
  def test_checkpointing_using_grain_built_in_tools(
      self,
      num_observations: int,
      split_full_length_features: bool,
  ):
    experimental.assert_equal_output_after_checkpoint(
        packing_concat_then_split.ConcatThenSplitIterDataset(
            self.dummy_iter_dataset(num_observations=num_observations),
            length_struct={"observation": 8, "index": 6},
            meta_features={"index"},
            split_full_length_features=split_full_length_features,
        )
    )

  @parameterized.product(
      bos_handling=list(BOSHandling),
  )
  def test_pack_sequence_longer_than_sequence_length(self, bos_handling):
    sequence_length = 10
    if bos_handling == BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS:
      bos_token_id = 1000
      bos_features = {"observation"}
    else:
      bos_token_id = None
      bos_features = {}
    ds = dataset.MapDataset.source([
        {"observation": np.repeat(1, 100)},  # 100 > sequence_length
    ]).to_iter_dataset()
    ds = packing_concat_then_split.ConcatThenSplitIterDataset(
        ds,
        length_struct={"observation": sequence_length},
        split_full_length_features=False,
        bos_handling=bos_handling,
        bos_token_id=bos_token_id,
        bos_features=bos_features,
    )
    with self.assertRaisesWithPredicateMatch(
        exceptions.PyGrainInternalError,
        lambda _: "Feature 'observation' has 100 tokens",
    ):
      next(iter(ds))

  def assert_equal_elements(
      self,
      actual_elements: list[dict[str, np.ndarray]],
      expected_elements: list[dict[str, np.ndarray]],
  ):
    try:
      np.testing.assert_equal(actual_elements, expected_elements)
    except AssertionError as e:
      raise AssertionError(
          f"Actual:\n{_print_elements(actual_elements)}\n\nExpected:\n{_print_elements(expected_elements)}"
      ) from e


if __name__ == "__main__":
  absltest.main()
