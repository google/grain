"""Tools for testing packing."""

# Unit test methods.
# pylint:disable=missing-class-docstring,missing-function-docstring

from collections.abc import Sequence
from typing import Any, Type
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset.transformations import packing
from grain._src.python.dataset.transformations import source
from jax import numpy as jnp
import numpy as np
import tree


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


def _common_test_body(
    packer_cls: Type[packing.PackIterDataset],
    input_elements,
    expected_elements,
    length_struct,
    *,
    num_packing_bins: int,
    seed: int = 0,
    shuffle_bins: bool = False,
    shuffle_bins_group_by_feature: str | None = None,
    meta_features: Sequence[str] = (),
    convert_input_to_np: bool = True,
    pack_alignment_struct: Any = None,
    padding_struct: Any = None,
    max_sequences_per_bin: int | None = None,
    kwargs: dict[str, Any] | None = None,
):
  """Factor out common test operations in a separate function."""
  if convert_input_to_np:
    input_elements = [
        {k: np.asarray(v) for k, v in d.items()} for d in input_elements
    ]
  expected_elements = [
      {k: np.asarray(v) for k, v in d.items()} for d in expected_elements
  ]
  ds = packer_cls(
      source.SourceMapDataset(input_elements).to_iter_dataset(),
      num_packing_bins=num_packing_bins,
      length_struct=length_struct,
      seed=seed,
      shuffle_bins=shuffle_bins,
      shuffle_bins_group_by_feature=shuffle_bins_group_by_feature,
      meta_features=meta_features,
      pack_alignment_struct=pack_alignment_struct,
      padding_struct=padding_struct,
      max_sequences_per_bin=max_sequences_per_bin,
      **(kwargs if kwargs else {}),
  )
  actual_elements = list(ds)
  np.testing.assert_equal(len(actual_elements), len(expected_elements))
  _assert_trees_equal(actual_elements, expected_elements)


class BaseFirstFitPackIterDatasetTest(parameterized.TestCase):
  """Base test for FirstFitPackIterDataset.

  This class contains the test suite for the First-Fit packing algorithm.
  It also serves as a base for other packing algorithm tests (e.g., BestFit),
  which can inherit from it to reuse general packing test cases.
  """

  packer_cls = packing.FirstFitPackIterDataset
  kwargs = {}

  def test_packing_scenario_1(self):
    """Provides the foundational comparison case for the two algorithms."""
    # With bins of capacity 10 and items [4, 8, 2]:
    # 1. Item 4 is placed in Bin 0. (Bin 0 has 6 space left).
    # 2. Item 8 can't fit in Bin 0, so it's placed in Bin 1.
    # 3. Item 2 arrives. First-Fit starts searching from the beginning and
    #    places it in the *first* available spot: Bin 0.
    input_elements = [
        {"features": [1, 1, 1, 1]},
        {"features": [2, 2, 2, 2, 2, 2, 2, 2]},
        {"features": [3, 3]},
    ]
    length_struct = {"features": 10}
    expected_elements = [
        {
            "features": ([1] * 4) + ([3] * 2) + ([0] * 4),
            "features_segment_ids": ([1] * 4) + ([2] * 2) + ([0] * 4),
            "features_positions": list(range(4)) + list(range(2)) + ([0] * 4),
        },
        {
            "features": ([2] * 8) + ([0] * 2),
            "features_segment_ids": ([1] * 8) + ([0] * 2),
            "features_positions": list(range(8)) + ([0] * 2),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=2,
    )

  def test_packing_scenario_2(self):
    """Scenario: An item can fit in two bins with different amounts of free space."""
    # First-Fit places the item of size 4 in the first available bin (Bin 0),
    # even though Bin 1 would be a tighter fit.
    input_elements = [
        {"features": [1] * 10},
        {"features": [2] * 15},
        {"features": [3] * 4},
    ]
    length_struct = {"features": 20}
    expected_elements = [
        {
            "features": ([1] * 10) + ([3] * 4) + ([0] * 6),
            "features_segment_ids": ([1] * 10) + ([2] * 4) + ([0] * 6),
            "features_positions": list(range(10)) + list(range(4)) + ([0] * 6),
        },
        {
            "features": ([2] * 15) + ([0] * 5),
            "features_segment_ids": ([1] * 15) + ([0] * 5),
            "features_positions": list(range(15)) + ([0] * 5),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=2,
    )

  def test_packing_scenario_3(self):
    """Scenario where FF is forced to use an extra bin for the final item."""
    # 1. Item 50 -> Bin 0. (rem: 50)
    # 2. Item 70 -> Bin 1. (rem: 30)
    # 3. Item 25 -> Bin 0. (rem: 25)
    # 4. Item 40 arrives. It can't fit in Bin 0 (rem 25) or Bin 1 (rem 30).
    #    It is forced to open a new, third bin.
    input_elements = [
        {"features": [1] * 50},
        {"features": [2] * 70},
        {"features": [3] * 25},
        {"features": [4] * 40},
    ]
    length_struct = {"features": 100}
    expected_elements = [
        {
            "features": ([1] * 50) + ([3] * 25) + ([0] * 25),
            "features_segment_ids": ([1] * 50) + ([2] * 25) + ([0] * 25),
            "features_positions": (
                list(range(50)) + list(range(25)) + ([0] * 25)
            ),
        },
        {
            "features": ([2] * 70) + ([0] * 30),
            "features_segment_ids": ([1] * 70) + ([0] * 30),
            "features_positions": list(range(70)) + ([0] * 30),
        },
        {
            "features": ([4] * 40) + ([0] * 60),
            "features_segment_ids": ([1] * 40) + ([0] * 60),
            "features_positions": list(range(40)) + ([0] * 60),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=3,
    )

  def test_packing_scenario_4(self):
    """Scenario where FF's prior choices prevent fitting a final large item."""
    # After packing {10, 4} into Bin 0 and {15} into Bin 1, the largest
    # remaining empty space is 6. The final item of size 10 cannot fit
    # and is forced to open a new, third bin.
    input_elements = [
        {"features": [1] * 10},
        {"features": [2] * 15},
        {"features": [3] * 4},
        {"features": [4] * 10},
    ]
    length_struct = {"features": 20}
    expected_elements = [
        {
            "features": ([1] * 10) + ([3] * 4) + ([0] * 6),
            "features_segment_ids": ([1] * 10) + ([2] * 4) + ([0] * 6),
            "features_positions": list(range(10)) + list(range(4)) + ([0] * 6),
        },
        {
            "features": ([2] * 15) + ([0] * 5),
            "features_segment_ids": ([1] * 15) + ([0] * 5),
            "features_positions": list(range(15)) + ([0] * 5),
        },
        {
            "features": ([4] * 10) + ([0] * 10),
            "features_segment_ids": ([1] * 10) + ([0] * 10),
            "features_positions": list(range(10)) + ([0] * 10),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=3,
    )

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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=num_packing_bins,
    )

  def test_bfloat16(self):
    input_elements = [
        {
            "soft_tokens": np.asarray(
                [
                    [[0.1, 0.2, 0.3]],
                    [[0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9]],
                ],
                dtype=jnp.bfloat16,
            ),
        },
        {
            "soft_tokens": np.asarray(
                [
                    [[1.1, 1.2, 1.3]],
                    [[1.4, 1.5, 1.6]],
                    [[1.7, 1.8, 1.9]],
                ],
                dtype=jnp.bfloat16,
            ),
        },
        {
            "soft_tokens": np.asarray(
                [
                    [[2.1, 2.2, 2.3]],
                    [[2.4, 2.5, 2.6]],
                    [[2.7, 2.8, 2.9]],
                ],
                dtype=jnp.bfloat16,
            ),
        },
    ]

    length_struct = {"soft_tokens": 4}

    expected_elements = [
        {
            "soft_tokens": np.asarray(
                [
                    [[0.1, 0.2, 0.3]],
                    [[0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9]],
                    [[0.0, 0.0, 0.0]],
                ],
                dtype=jnp.bfloat16,
            ),
            "soft_tokens_positions": [0, 1, 2, 0],
            "soft_tokens_segment_ids": [1, 1, 1, 0],
        },
        {
            "soft_tokens": np.asarray(
                [
                    [[1.1, 1.2, 1.3]],
                    [[1.4, 1.5, 1.6]],
                    [[1.7, 1.8, 1.9]],
                    [[0.0, 0.0, 0.0]],
                ],
                dtype=jnp.bfloat16,
            ),
            "soft_tokens_positions": [0, 1, 2, 0],
            "soft_tokens_segment_ids": [1, 1, 1, 0],
        },
        {
            "soft_tokens": np.asarray(
                [
                    [[2.1, 2.2, 2.3]],
                    [[2.4, 2.5, 2.6]],
                    [[2.7, 2.8, 2.9]],
                    [[0.0, 0.0, 0.0]],
                ],
                dtype=jnp.bfloat16,
            ),
            "soft_tokens_positions": [0, 1, 2, 0],
            "soft_tokens_segment_ids": [1, 1, 1, 0],
        },
    ]

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=3,
    )

  def test_missing_length_struct_feature(self):
    input_elements = [
        {
            "a": np.asarray([1, 2, 3]),
            # This feature is not in the length struct so it should be ignored.
            "b": np.asarray([1, 2, 3]),
        },
    ]
    length_struct = {"a": 3}

    ld = self.packer_cls(
        source.SourceMapDataset(input_elements).to_iter_dataset(),
        num_packing_bins=1,
        length_struct=length_struct,
        **self.kwargs,
    )
    result = next(iter(ld))
    np.testing.assert_array_equal(result["a"], np.asarray([1, 2, 3]))
    np.testing.assert_array_equal(
        result["a_segment_ids"], np.asarray([1, 1, 1])
    )
    np.testing.assert_array_equal(result["a_positions"], np.asarray([0, 1, 2]))

  def test_variable_key_features(self):
    input_elements = [
        {
            "a": np.asarray([1, 2, 3]),
            "b": np.asarray([1, 2, 3]),
        },
        # This element is missing the "b" feature, so we should raise an error.
        {
            "a": np.asarray([1, 2, 3]),
        },
    ]

    length_struct = {"a": 3, "b": 3}
    ld = self.packer_cls(
        source.SourceMapDataset(input_elements).to_iter_dataset(),
        num_packing_bins=1,
        length_struct=length_struct,
        **self.kwargs,
    )
    with self.assertRaisesRegex(Exception, "'b'"):
      next(iter(ld))

  @parameterized.parameters(
      {"num_packing_bins": 3},
      {"num_packing_bins": 5},
  )
  def test_pack_sequences_length_shuffle_bins(
      self,
      num_packing_bins: int,
  ):
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=num_packing_bins,
        kwargs=self.kwargs,
        shuffle_bins=True,
    )

  @parameterized.product(
      num_packing_bins=[4, 5],
  )
  def test_pack_sequences_length_epoch_aware_shuffle_bins(
      self, num_packing_bins: int
  ):
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "epoch": [2],
        },
        {
            "inputs": [4, 5],
            "epoch": [3],
        },
        {
            "inputs": [6],
            "epoch": [3],
        },
        {
            "inputs": [7],
            "epoch": [2],
        },
        {
            "inputs": [8, 9, 10],
            "epoch": [2],
        },
    ]

    length_struct = {"inputs": 3, "epoch": 3}

    expected_elements = [
        {
            "inputs": [7, 0, 0],
            "epoch": [2, 0, 0],
            "inputs_segment_ids": [1, 0, 0],
            "inputs_positions": [0, 0, 0],
        },
        {
            "inputs": [1, 2, 3],
            "epoch": [2, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
        {
            "inputs": [8, 9, 10],
            "epoch": [2, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
        {
            "inputs": [4, 5, 6],
            "epoch": [3, 3, 0],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
    ]

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=num_packing_bins,
        shuffle_bins=True,
        shuffle_bins_group_by_feature="epoch",
        meta_features=["epoch"],
        kwargs=self.kwargs,
    )

  def test_pack_sequences_length_epoch_aware_shuffle_bins_with_epoch0(self):
    # This test guards against divide by zero errors.
    input_elements = [
        {
            "inputs": [1, 2, 3],
            "epoch": [0],
        },
        {
            "inputs": [4, 5],
            "epoch": [0],
        },
        {
            "inputs": [6],
            "epoch": [0],
        },
        {
            "inputs": [7],
            "epoch": [0],
        },
    ]

    length_struct = {"inputs": 3, "epoch": 3}

    expected_elements = [
        {
            "inputs": [4, 5, 6],
            "epoch": [0, 0, 0],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
        {
            "inputs": [7, 0, 0],
            "epoch": [0, 0, 0],
            "inputs_segment_ids": [1, 0, 0],
            "inputs_positions": [0, 0, 0],
        },
        {
            "inputs": [1, 2, 3],
            "epoch": [0, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
    ]

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=5,
        shuffle_bins=True,
        shuffle_bins_group_by_feature="epoch",
        meta_features=["epoch"],
        kwargs=self.kwargs,
    )

  @parameterized.parameters(
      {"seed": 0},
      {"seed": 42},
      {"seed": 123},
  )
  def test_shuffle_seed_default(self, seed):
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

    element_1 = {
        "inputs": [6, 0, 0],
        "targets": [50, 60, 0],
        "inputs_segment_ids": [1, 0, 0],
        "targets_segment_ids": [1, 1, 0],
        "inputs_positions": [0, 0, 0],
        "targets_positions": [0, 1, 0],
    }
    element_2 = {
        "inputs": [1, 2, 3],
        "targets": [10, 0, 0],
        "inputs_segment_ids": [1, 1, 1],
        "targets_segment_ids": [1, 0, 0],
        "inputs_positions": [0, 1, 2],
        "targets_positions": [0, 0, 0],
    }
    element_3 = {
        "inputs": [4, 5, 0],
        "targets": [20, 30, 40],
        "inputs_segment_ids": [1, 1, 0],
        "targets_segment_ids": [1, 1, 1],
        "inputs_positions": [0, 1, 0],
        "targets_positions": [0, 1, 2],
    }
    if seed == 0:
      expected_elements = [element_1, element_2, element_3]
    elif seed == 42:
      expected_elements = [element_1, element_3, element_2]
    elif seed == 123:
      expected_elements = [element_2, element_1, element_3]
    else:
      raise ValueError(f"Unexpected seed: {seed}")

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=5,
        kwargs=self.kwargs,
        seed=seed,
        shuffle_bins=True,
    )

  # Don't convert epoch `1` to `np.asarray(1)` during testing
  # to verify how packing deals with raw integers.
  def test_raw_ints(self):
    input_elements = [
        {
            "inputs": np.asarray([1, 2, 3]),
            "epoch": 1,
        },
        {
            "inputs": np.asarray([4, 5]),
            "epoch": 2,
        },
        {
            "inputs": np.asarray([6]),
            "epoch": 3,
        },
    ]

    length_struct = {"inputs": 3, "epoch": 3}

    expected_elements = [
        {
            "inputs": [1, 2, 3],
            "epoch": [1, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "inputs_positions": [0, 1, 2],
        },
        {
            "inputs": [4, 5, 6],
            "epoch": [2, 3, 0],
            "inputs_segment_ids": [1, 1, 2],
            "inputs_positions": [0, 1, 0],
        },
    ]

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=5,
        convert_input_to_np=False,
        meta_features=["epoch"],
        kwargs=self.kwargs,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=2,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=2,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=2,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=1,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=3,
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
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=2,
    )

  @parameterized.product(
      convert_input_to_np=[True, False],
  )
  def test_meta_features(self, convert_input_to_np: bool):
    input_elements = [
        {
            "inputs": np.asarray([1, 2, 3]),
            "targets": np.asarray([10]),
            "meta_feature": 3,
        },
        {
            "inputs": np.asarray([4, 5]),
            "targets": np.asarray([20, 30, 40]),
            "meta_feature": 7,
        },
        {
            "inputs": np.asarray([6]),
            "targets": np.asarray([50, 60]),
            "meta_feature": 5,
        },
    ]
    length_struct = {"inputs": 3, "targets": 4, "meta_feature": 3}

    expected_elements = [
        {
            "inputs": [1, 2, 3],
            "targets": [10, 0, 0, 0],
            "inputs_segment_ids": [1, 1, 1],
            "targets_segment_ids": [1, 0, 0, 0],
            "inputs_positions": [0, 1, 2],
            "targets_positions": [0, 0, 0, 0],
            "meta_feature": [3, 0, 0],
        },
        {
            "inputs": [4, 5, 0],
            "targets": [20, 30, 40, 0],
            "inputs_segment_ids": [1, 1, 0],
            "targets_segment_ids": [1, 1, 1, 0],
            "inputs_positions": [0, 1, 0],
            "targets_positions": [0, 1, 2, 0],
            "meta_feature": [7, 0, 0],
        },
        {
            "inputs": [6, 0, 0],
            "targets": [50, 60, 0, 0],
            "inputs_segment_ids": [1, 0, 0],
            "targets_segment_ids": [1, 1, 0, 0],
            "inputs_positions": [0, 0, 0],
            "targets_positions": [0, 1, 0, 0],
            "meta_feature": [5, 0, 0],
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=3,
        meta_features=["meta_feature"],
        convert_input_to_np=convert_input_to_np,
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
    ld = self.packer_cls(
        source.SourceMapDataset(input_elements).to_iter_dataset(),
        num_packing_bins=2,
        length_struct=length_struct,
        shuffle_bins=True,
        **self.kwargs if self.kwargs else {},
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

    self.assertEqual(len(actual_elements), len(expected_elements))
    _assert_trees_equal(actual_elements, expected_elements)

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
    ld = self.packer_cls(
        source.SourceMapDataset(elements).repeat().to_iter_dataset(),
        num_packing_bins=4,
        length_struct=dict(row=100),
        shuffle_bins=shuffle_bins,
        **self.kwargs if self.kwargs else {},
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
    ld = self.packer_cls(
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

  @parameterized.named_parameters(
      # pack_alignment=1 should behave the same as no alignment.
      dict(
          testcase_name="1",
          pack_alignment=1,
          expected_elements=[
              {
                  "inputs": [1, 2, 3, 4, 5, 6, 0, 0],
                  "targets": [10, 20, 30, 40, 50, 60, -1, -1],
                  "inputs_segment_ids": [1, 1, 2, 2, 2, 3, 0, 0],
                  "targets_segment_ids": [1, 2, 2, 3, 3, 3, 0, 0],
                  "inputs_positions": [0, 1, 0, 1, 2, 0, 0, 0],
                  "targets_positions": [0, 0, 1, 0, 1, 2, 0, 0],
              },
          ],
      ),
      dict(
          testcase_name="4",
          pack_alignment=4,
          expected_elements=[
              {
                  "inputs": [1, 2, 0, 0, 3, 4, 5, 0],
                  "targets": [10, -1, -1, -1, 20, 30, -1, -1],
                  "inputs_segment_ids": [1, 1, 0, 0, 2, 2, 2, 0],
                  "targets_segment_ids": [1, 0, 0, 0, 2, 2, 0, 0],
                  "inputs_positions": [0, 1, 0, 0, 0, 1, 2, 0],
                  "targets_positions": [0, 0, 0, 0, 0, 1, 0, 0],
              },
              {
                  "inputs": [6, 0, 0, 0, 0, 0, 0, 0],
                  "targets": [40, 50, 60, -1, -1, -1, -1, -1],
                  "inputs_segment_ids": [1, 0, 0, 0, 0, 0, 0, 0],
                  "targets_segment_ids": [1, 1, 1, 0, 0, 0, 0, 0],
                  "inputs_positions": [0, 0, 0, 0, 0, 0, 0, 0],
                  "targets_positions": [0, 1, 2, 0, 0, 0, 0, 0],
              },
          ],
      ),
  )
  def test_pack_alignment_and_padding_struct(
      self, pack_alignment: int, expected_elements: list[dict[str, Any]]
  ):
    input_elements = [
        {"inputs": [1, 2], "targets": [10]},
        {"inputs": [3, 4, 5], "targets": [20, 30]},
        {"inputs": [6], "targets": [40, 50, 60]},
    ]
    length_struct = {"inputs": 8, "targets": 8}
    padding_struct = {"inputs": 0, "targets": -1}
    num_packing_bins = 2
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        pack_alignment_struct=tree.map_structure(
            lambda x: pack_alignment, length_struct
        ),
        padding_struct=padding_struct,
        num_packing_bins=num_packing_bins,
    )

  def test_pack_max_sequences_per_bin(self):
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

    expected_elements = [
        {
            "inputs": [1, 2, 3, 4, 5, 0],
            "targets": [10, 20, 30, 40, 0, 0],
            "inputs_segment_ids": [1, 1, 1, 2, 2, 0],
            "targets_segment_ids": [1, 2, 2, 2, 0, 0],
            "inputs_positions": [0, 1, 2, 0, 1, 0],
            "targets_positions": [0, 0, 1, 2, 0, 0],
        },
        {
            "inputs": [6, 0, 0, 0, 0, 0],
            "targets": [50, 60, 0, 0, 0, 0],
            "inputs_segment_ids": [1, 0, 0, 0, 0, 0],
            "targets_segment_ids": [1, 1, 0, 0, 0, 0],
            "inputs_positions": [0, 0, 0, 0, 0, 0],
            "targets_positions": [0, 1, 0, 0, 0, 0],
        },
    ]

    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        kwargs=self.kwargs,
        num_packing_bins=2,
        max_sequences_per_bin=2,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="Zero",
          max_sequences_per_bin=0,
      ),
      dict(
          testcase_name="Negative",
          max_sequences_per_bin=-1,
      ),
  )
  def test_pack_max_sequences_per_bin_invalid_value(
      self, max_sequences_per_bin
  ):
    with self.assertRaisesRegex(
        ValueError,
        "If specified, `max_sequences_per_bin` must be a positive integer,"
        f" but got {max_sequences_per_bin}.",
    ):
      self.packer_cls(
          source.SourceMapDataset([]).to_iter_dataset(),
          length_struct={"inputs": 6, "targets": 6},
          num_packing_bins=2,
          max_sequences_per_bin=max_sequences_per_bin,
      )


class BaseBestFitPackIterDatasetTest(BaseFirstFitPackIterDatasetTest):
  """Base test for the Best-Fit packing algorithm.

  This class inherits from BaseFirstFitPackIterDatasetTest to reuse all
  general packing tests (e.g., for checkpointing, data types). It overrides
  only the packing scenario tests that are specific to the Best-Fit strategy.
  """

  packer_cls = packing.BestFitPackIterDataset

  def test_packing_scenario_1(self):
    """Provides the foundational comparison case for the two algorithms."""
    # With bins of capacity 10 and items [4, 8, 2]:
    # 1. Item 4 is placed in Bin 0. (Bin 0 has 6 space left).
    # 2. Item 8 is placed in Bin 1. (Bin 1 has 2 space left).
    # 3. Item 2 arrives. Best-Fit checks all options and finds that Bin 1 is
    #    the "best" (tightest) fit, as it leaves 0 empty space.
    input_elements = [
        {"features": [1, 1, 1, 1]},
        {"features": [2, 2, 2, 2, 2, 2, 2, 2]},
        {"features": [3, 3]},
    ]
    length_struct = {"features": 10}
    expected_elements = [
        {
            "features": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "features_segment_ids": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "features_positions": [0, 1, 2, 3, 0, 0, 0, 0, 0, 0],
        },
        {
            "features": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
            "features_segment_ids": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            "features_positions": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1],
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=2,
    )

  def test_packing_scenario_2(self):
    """Scenario: An item can fit in two bins with different amounts of free space."""
    # Best-Fit checks both bins and places the item of size 4 in Bin 1,
    # as it results in the least leftover space (1 vs 6).
    input_elements = [
        {"features": [1] * 10},
        {"features": [2] * 15},
        {"features": [3] * 4},
    ]
    length_struct = {"features": 20}
    expected_elements = [
        {
            "features": ([1] * 10) + ([0] * 10),
            "features_segment_ids": ([1] * 10) + ([0] * 10),
            "features_positions": list(range(10)) + ([0] * 10),
        },
        {
            "features": ([2] * 15) + ([3] * 4) + ([0] * 1),
            "features_segment_ids": ([1] * 15) + ([2] * 4) + ([0] * 1),
            "features_positions": list(range(15)) + list(range(4)) + ([0] * 1),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=2,
    )

  def test_packing_scenario_3(self):
    """Scenario where BF's smarter choices result in using fewer bins."""
    # 1. Item 50 -> Bin 0. (rem: 50)
    # 2. Item 70 -> Bin 1. (rem: 30)
    # 3. Item 25 -> Bin 1 (best fit, leaves 5). (rem: Bin 0: 50, Bin 1: 5)
    # 4. Item 40 arrives. It fits nicely into Bin 0 (best fit, leaves 10).
    #    The result is a compact packing into only 2 bins.
    input_elements = [
        {"features": [1] * 50},
        {"features": [2] * 70},
        {"features": [3] * 25},
        {"features": [4] * 40},
    ]
    length_struct = {"features": 100}
    expected_elements = [
        {
            "features": ([1] * 50) + ([4] * 40) + ([0] * 10),
            "features_segment_ids": ([1] * 50) + ([2] * 40) + ([0] * 10),
            "features_positions": (
                list(range(50)) + list(range(40)) + ([0] * 10)
            ),
        },
        {
            "features": ([2] * 70) + ([3] * 25) + ([0] * 5),
            "features_segment_ids": ([1] * 70) + ([2] * 25) + ([0] * 5),
            "features_positions": list(range(70)) + list(range(25)) + ([0] * 5),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=3,
    )

  def test_packing_scenario_4(self):
    """Scenario where BF's prior choices allow fitting a final large item."""
    # After packing {10} into Bin 0 and {15, 4} into Bin 1, Best-Fit
    # strategically preserved a large empty space of 10 in Bin 0. The final
    # item fits perfectly into this space, allowing for a compact 2-bin pack.
    input_elements = [
        {"features": [1] * 10},
        {"features": [2] * 15},
        {"features": [3] * 4},
        {"features": [4] * 10},
    ]
    length_struct = {"features": 20}
    expected_elements = [
        {
            "features": ([1] * 10) + ([4] * 10),
            "features_segment_ids": ([1] * 10) + ([2] * 10),
            "features_positions": list(range(10)) + list(range(10)),
        },
        {
            "features": ([2] * 15) + ([3] * 4) + ([0] * 1),
            "features_segment_ids": ([1] * 15) + ([2] * 4) + ([0] * 1),
            "features_positions": list(range(15)) + list(range(4)) + ([0] * 1),
        },
    ]
    _common_test_body(
        self.packer_cls,
        input_elements,
        expected_elements,
        length_struct,
        num_packing_bins=3,
    )


if __name__ == "__main__":
  absltest.main()
