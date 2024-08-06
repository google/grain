"""Unit tests for ConcatThenSplitIterDataset."""

from collections.abc import Set
from typing import Any, cast

from absl.testing import absltest
from absl.testing import parameterized
import chex
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import concat_then_split
from grain._src.python.dataset.transformations import source
import numpy as np


def _make_single_repeated_feature(
    index: int, value: int
) -> dict[str, np.ndarray | int]:
  return {
      "observation": np.repeat(value, value),
      "index": index + 1,
  }


# Pretty print elements for debugging.
def pprint_elements(elements: list[dict[str, np.ndarray]]):
  s = "\n"
  for element in elements:
    elements_str = "\n".join(
        [f'   "{k}": np.asarray({list(v)}),' for k, v in element.items()]
    )
    s += f"{{\n{elements_str}\n}},\n"
  return s


class ConcatThenSplitIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # observations will be [
    #   [1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5],
    #   [6, 6, 6, 6, 6, 6], [1], [2, 2], [3, 3, 3], ...
    # ].
    self.test_ds = (
        source.RangeMapDataset(1, 7)
        .repeat()
        .map_with_index(_make_single_repeated_feature)
    )

  def test_meta_features_not_restricting(self):
    # Pack 9 elements.
    ds = concat_then_split.ConcatThenSplitIterDataset(
        self.test_ds[:9].to_iter_dataset(),
        sequence_lengths={"observation": 6, "index": 6},
        meta_features={"index"},
    )
    actual_elements = list(ds)
    chex.assert_trees_all_equal(
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
            # Fully packed comes first before element 5 continues.
            {
                "observation": np.asarray([6, 6, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 1]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 5]),
                "index": np.asarray([6, 0, 0, 0, 0, 0]),
            },
            {
                "observation": np.asarray([5, 5, 5, 1, 2, 2]),
                "observation_segment_ids": np.asarray([1, 1, 1, 2, 3, 3]),
                "observation_positions": np.asarray([0, 1, 2, 0, 0, 1]),
                "index": np.asarray([5, 7, 8, 0, 0, 0]),
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
    ds = concat_then_split.ConcatThenSplitIterDataset(
        self.test_ds[:9].to_iter_dataset(),
        sequence_lengths={"observation": 6, "index": 2},
        meta_features={"index"},
    )
    actual_elements = list(ds)
    chex.assert_trees_all_equal(
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
                "observation": np.asarray([6, 6, 6, 6, 6, 6]),
                "observation_segment_ids": np.asarray([1, 1, 1, 1, 1, 1]),
                "observation_positions": np.asarray([0, 1, 2, 3, 4, 5]),
                "index": np.asarray([6, 0]),
            },
            {
                "observation": np.asarray([4, 5, 5, 5, 5, 5]),
                "observation_segment_ids": np.asarray([1, 2, 2, 2, 2, 2]),
                "observation_positions": np.asarray([0, 0, 1, 2, 3, 4]),
                "index": np.asarray([4, 5]),
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

  def test_additional_sequence_length(self):
    ds = concat_then_split.ConcatThenSplitIterDataset(
        self.test_ds.to_iter_dataset(),
        sequence_lengths={"observation": 6, "index": 6, "does_not_exist": 6},
        meta_features={"index"},
    )
    with self.assertRaisesRegex(ValueError, "Parent element has structure"):
      list(ds)

  def test_missingsequence_length(self):
    ds = concat_then_split.ConcatThenSplitIterDataset(
        self.test_ds.to_iter_dataset(),
        sequence_lengths={"index": 6},
        meta_features={"index"},
    )
    with self.assertRaisesRegex(ValueError, "Parent element has structure"):
      list(ds)

  def test_bos_id_not_provided(self):
    with self.assertRaisesRegex(
        ValueError, "bos_id must be set if insert_bos_after_split is True."
    ):
      concat_then_split.ConcatThenSplitIterDataset(
          self.test_ds.to_iter_dataset(),
          sequence_lengths={"observation": 6, "index": 6},
          meta_features={"index"},
          insert_bos_after_split=True,
      )
    with self.assertRaisesRegex(
        ValueError,
        "bos_id must be set if replace_with_bos_after_split is True.",
    ):
      concat_then_split.ConcatThenSplitIterDataset(
          self.test_ds.to_iter_dataset(),
          sequence_lengths={"observation": 6, "index": 6},
          meta_features={"index"},
          replace_with_bos_after_split=True,
      )

  @parameterized.product(
      insert_bos_after_split=[True, False],
      replace_with_bos_after_split=[True, False],
  )
  def test_replace_with_bos_after_split(
      self, insert_bos_after_split: bool, replace_with_bos_after_split: bool
  ):
    bos_id = 999
    elements = [
        {
            "index": 1,
            "seed": [1, 1001],
            "observation": [34, 2, 32],
        },
        {
            "index": 2,
            "seed": [2, 2002],
            "observation": [2, 49, 99, 5],
        },
        {
            "index": 3,
            "seed": [3, 3003],
            "observation": [2, 3, 5],
        },
        {
            "index": 4,
            "seed": [4, 4004],
            "observation": [2, 8],
        },
        {
            "index": 5,
            "seed": [5, 5005],
            "observation": [76, 42],
        },
        {
            "index": 6,
            "seed": [6, 6006],
            "observation": [1, 2, 3, 4, 5],
        },
    ]
    ds = source.SourceMapDataset(elements)
    sequence_lengths = {
        "index": 5,
        "seed": 5,
        "observation": 5,
    }
    if insert_bos_after_split and replace_with_bos_after_split:
      with self.assertRaisesRegex(
          ValueError,
          "insert_bos_after_split and replace_with_bos_after_split cannot both"
          " be True at the same time.",
      ):
        concat_then_split.ConcatThenSplitIterDataset(
            ds.to_iter_dataset(),
            sequence_lengths=sequence_lengths,
            meta_features={"index", "seed"},
            insert_bos_after_split=insert_bos_after_split,
            replace_with_bos_after_split=replace_with_bos_after_split,
            bos_id=bos_id,
        )
      return
    ds = concat_then_split.ConcatThenSplitIterDataset(
        ds.to_iter_dataset(),
        sequence_lengths=sequence_lengths,
        meta_features={"index", "seed"},
        insert_bos_after_split=insert_bos_after_split,
        replace_with_bos_after_split=replace_with_bos_after_split,
        bos_id=bos_id,
    )
    actual_elements = list(ds)
    if insert_bos_after_split:
      chex.assert_trees_all_equal(
          actual_elements,
          [
              {
                  "index": np.asarray([1, 2, 0, 0, 0]),
                  "seed": np.asarray([1, 1001, 2, 2002, 0]),
                  "observation": np.asarray([34, 2, 32, 2, 49]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 2, 0, 1]),
              },
              {
                  "index": np.asarray([2, 3, 0, 0, 0]),
                  "seed": np.asarray([2, 2002, 3, 3003, 0]),
                  # BOS inserted.
                  "observation": np.asarray([999, 99, 5, 2, 3]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 2, 0, 1]),
              },
              {
                  "index": np.asarray([3, 4, 0, 0, 0]),
                  "seed": np.asarray([3, 3003, 4, 4004, 0]),
                  # Only inserting BOS doesn't make sense.
                  "observation": np.asarray([999, 5, 2, 8, 0]),
                  "observation_segment_ids": np.asarray([1, 1, 2, 2, 0]),
                  "observation_positions": np.asarray([0, 1, 0, 1, 0]),
              },
              {
                  "index": np.asarray([6, 0, 0, 0, 0]),
                  "seed": np.asarray([6, 6006, 0, 0, 0]),
                  "observation": np.asarray([1, 2, 3, 4, 5]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 1, 1]),
                  "observation_positions": np.asarray([0, 1, 2, 3, 4]),
              },
              {
                  "index": np.asarray([5, 0, 0, 0, 0]),
                  "seed": np.asarray([5, 5005, 0, 0, 0]),
                  "observation": np.asarray([76, 42, 0, 0, 0]),
                  "observation_segment_ids": np.asarray([1, 1, 0, 0, 0]),
                  "observation_positions": np.asarray([0, 1, 0, 0, 0]),
              },
          ],
      )
    elif replace_with_bos_after_split:
      chex.assert_trees_all_equal(
          actual_elements,
          [
              {
                  "index": np.asarray([1, 2, 0, 0, 0]),
                  "seed": np.asarray([1, 1001, 2, 2002, 0]),
                  "observation": np.asarray([34, 2, 32, 2, 49]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 2, 0, 1]),
              },
              {
                  "index": np.asarray([2, 3, 0, 0, 0]),
                  "seed": np.asarray([2, 2002, 3, 3003, 0]),
                  "observation": np.asarray([999, 5, 2, 3, 5]),
                  "observation_segment_ids": np.asarray([1, 1, 2, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 0, 1, 2]),
              },
              {
                  "index": np.asarray([6, 0, 0, 0, 0]),
                  "seed": np.asarray([6, 6006, 0, 0, 0]),
                  "observation": np.asarray([1, 2, 3, 4, 5]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 1, 1]),
                  "observation_positions": np.asarray([0, 1, 2, 3, 4]),
              },
              {
                  "index": np.asarray([4, 5, 0, 0, 0]),
                  "seed": np.asarray([4, 4004, 5, 5005, 0]),
                  "observation": np.asarray([2, 8, 76, 42, 0]),
                  "observation_segment_ids": np.asarray([1, 1, 2, 2, 0]),
                  "observation_positions": np.asarray([0, 1, 0, 1, 0]),
              },
          ],
      )
    else:
      chex.assert_trees_all_equal(
          actual_elements,
          [
              {
                  "index": np.asarray([1, 2, 0, 0, 0]),
                  "seed": np.asarray([1, 1001, 2, 2002, 0]),
                  "observation": np.asarray([34, 2, 32, 2, 49]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 2, 0, 1]),
              },
              {
                  "index": np.asarray([2, 3, 0, 0, 0]),
                  "seed": np.asarray([2, 2002, 3, 3003, 0]),
                  "observation": np.asarray([99, 5, 2, 3, 5]),
                  "observation_segment_ids": np.asarray([1, 1, 2, 2, 2]),
                  "observation_positions": np.asarray([0, 1, 0, 1, 2]),
              },
              {
                  "index": np.asarray([6, 0, 0, 0, 0]),
                  "seed": np.asarray([6, 6006, 0, 0, 0]),
                  "observation": np.asarray([1, 2, 3, 4, 5]),
                  "observation_segment_ids": np.asarray([1, 1, 1, 1, 1]),
                  "observation_positions": np.asarray([0, 1, 2, 3, 4]),
              },
              {
                  "index": np.asarray([4, 5, 0, 0, 0]),
                  "seed": np.asarray([4, 4004, 5, 5005, 0]),
                  "observation": np.asarray([2, 8, 76, 42, 0]),
                  "observation_segment_ids": np.asarray([1, 1, 2, 2, 0]),
                  "observation_positions": np.asarray([0, 1, 0, 1, 0]),
              },
          ],
      )

  @parameterized.parameters(
      {"checkpoint_steps": frozenset()},
      {"checkpoint_steps": {2}},
      {"checkpoint_steps": {0, 3}},
      {"checkpoint_steps": {0, 1, 2, 3, 4, 5}},
  )
  def test_checkpointing(self, checkpoint_steps: Set[int]):
    def _create_iter(state: dict[str, Any] | None):
      ds = concat_then_split.ConcatThenSplitIterDataset(
          self.test_ds[:12].to_iter_dataset(),
          sequence_lengths={"observation": 8, "index": 6},
          meta_features={"index"},
      )
      ds_iter = cast(dataset.DatasetIterator, iter(ds))
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
      num_observations=list(range(12)), recreate_iter=[True, False]
  )
  def test_checkpointing_after_stop_iteration(
      self, num_observations: int, recreate_iter: bool
  ):
    def _create_iter(state: dict[str, Any] | None):
      ds = concat_then_split.ConcatThenSplitIterDataset(
          self.test_ds[:num_observations].to_iter_dataset(),
          sequence_lengths={"observation": 8, "index": 6},
          meta_features={"index"},
      )
      ds_iter = cast(dataset.DatasetIterator, iter(ds))
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

  def assert_equal_elements(
      self,
      actual_elements: list[dict[str, np.ndarray]],
      expected_elements: list[dict[str, np.ndarray]],
  ):
    try:
      chex.assert_trees_all_equal(actual_elements, expected_elements)
    except AssertionError as e:
      raise AssertionError(
          f"Actual:\n{pprint_elements(actual_elements)}\n\nExpected:\n{pprint_elements(expected_elements)}"
      ) from e


if __name__ == "__main__":
  absltest.main()
