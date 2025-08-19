"""Tests for packing.py."""

from absl.testing import absltest
from grain._src.core import tree_lib
from grain._src.python import record
from grain._src.python.experimental.example_packing import packing
import numpy as np


def create_input_dataset(input_dataset_elements):
  """Create a dataset from list of dict elements."""

  def _to_record(i, d):
    metadata = record.RecordMetadata(index=i)
    return record.Record(metadata, {k: np.array(v) for k, v in d.items()})

  return [_to_record(i, d) for (i, d) in enumerate(input_dataset_elements)]


def create_expected_dataset(values, segmentations, positions, indexes):
  dataset = []
  for value, segmentation, position, index in zip(
      values, segmentations, positions, indexes
  ):
    dataset.append(
        record.Record(
            data=(value, segmentation, position),
            metadata=record.RecordMetadata(index=index),
        )
    )
  return dataset


def common_test_body(
    input_dataset, expected_packed_dataset, length_struct, batch_size
):
  """Factor out common test operations in a separate function."""
  input_dataset = create_input_dataset(input_dataset)
  pack_op = packing.PackAndBatchOperation(
      batch_size=batch_size,
      length_struct=length_struct,
  )
  packed_dataset = pack_op(input_dataset)  # pytype: disable=wrong-arg-types
  actual_packed_dataset = list(packed_dataset)
  np.testing.assert_equal(
      len(actual_packed_dataset), len(expected_packed_dataset)
  )

  def _check_equivalence(path, actual_val, expected_val):
    np.testing.assert_array_equal(
        actual_val,
        expected_val,
        err_msg=(
            f"Pytrees differ at path {path}.\n\n"
            f"Actual: {actual_val}\n\nExpected: {expected_val}"
        ),
    )

  for actual_data, expected_data in zip(
      actual_packed_dataset, expected_packed_dataset
  ):
    np.testing.assert_equal(
        actual_data.metadata.index, expected_data.metadata.index
    )
    tree_lib.map_structure_with_path(
        _check_equivalence, actual_data.data, expected_data.data
    )


class PackingTest(absltest.TestCase):
  """Tests for the packed_dataset module."""

  # Begin adapted tests from http://shortn/_wlj2e83614
  def test_pack_sequences_length_3(self):
    input_dataset = [
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
    batch_size = 3

    expected_values = [{
        "inputs": np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]]),
        "targets": np.array([[10, 0, 0], [20, 30, 40], [50, 60, 0]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "targets": np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2], [0, 1, 0], [0, 0, 0]]),
        "targets": np.array([[0, 0, 0], [0, 1, 2], [0, 1, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_length_4(self):
    input_dataset = [
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
    batch_size = 2

    expected_values = [{
        "inputs": np.array([[1, 2, 3, 6], [4, 5, 0, 0]]),
        "targets": np.array([[10, 50, 60, 0], [20, 30, 40, 0]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1, 2], [1, 1, 0, 0]]),
        "targets": np.array([[1, 2, 2, 0], [1, 1, 1, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2, 0], [0, 1, 0, 0]]),
        "targets": np.array([[0, 0, 1, 0], [0, 1, 2, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_length_5(self):
    input_dataset = [
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
    batch_size = 2

    expected_values = [{
        "inputs": np.array([[1, 2, 3, 4, 5], [6, 0, 0, 0, 0]]),
        "targets": np.array([[10, 20, 30, 40, 0], [50, 60, 0, 0, 0]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1, 2, 2], [1, 0, 0, 0, 0]]),
        "targets": np.array([[1, 2, 2, 2, 0], [1, 1, 0, 0, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2, 0, 1], [0, 0, 0, 0, 0]]),
        "targets": np.array([[0, 0, 1, 2, 0], [0, 1, 0, 0, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_length_6(self):
    input_dataset = [
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
    batch_size = 1

    expected_values = [{
        "inputs": np.array([[1, 2, 3, 4, 5, 6]]),
        "targets": np.array([[10, 20, 30, 40, 50, 60]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1, 2, 2, 3]]),
        "targets": np.array([[1, 2, 2, 2, 3, 3]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2, 0, 1, 0]]),
        "targets": np.array([[0, 0, 1, 2, 0, 1]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_length_7(self):
    input_dataset = [
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
    batch_size = 1

    expected_values = [{
        "inputs": np.array([[1, 2, 3, 4, 5, 6, 0]]),
        "targets": np.array([[10, 20, 30, 40, 50, 60, 0]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1, 2, 2, 3, 0]]),
        "targets": np.array([[1, 2, 2, 2, 3, 3, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2, 0, 1, 0, 0]]),
        "targets": np.array([[0, 0, 1, 2, 0, 1, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_different_lengths(self):
    input_dataset = [
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
    batch_size = 3

    expected_values = [{
        "inputs": np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]]),
        "targets": np.array([[10, 0, 0, 0], [20, 30, 40, 0], [50, 60, 0, 0]]),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "targets": np.array([[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2], [0, 1, 0], [0, 0, 0]]),
        "targets": np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 1, 0, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )
    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_two_dimensional_features(self):
    input_dataset = [
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
    batch_size = 2

    expected_values = [{
        "input_tokens": np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]),
        "input_vectors": np.array([
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
        ]),
        "targets": np.array([[10, 0, 0, 0, 0], [20, 30, 40, 50, 60]]),
    }]
    expected_segmentations = [{
        "input_tokens": np.array([[1, 1, 1, 0, 0], [1, 1, 2, 0, 0]]),
        "input_vectors": np.array([[1, 1, 1], [1, 1, 2]]),
        "targets": np.array([[1, 0, 0, 0, 0], [1, 1, 1, 2, 2]]),
    }]
    expected_positions = [{
        "input_tokens": np.array([[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]]),
        "input_vectors": np.array([[0, 1, 2], [0, 1, 0]]),
        "targets": np.array([[0, 0, 0, 0, 0], [0, 1, 2, 0, 1]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  # End adpated tests from: http://shortn/_wlj2e83614

  def test_pack_sequences_into_multiple_batches(self):
    input_dataset = [
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
    batch_size = 2

    expected_values = [
        {
            "inputs": np.array([[1, 2, 3], [4, 5, 0]]),
            "targets": np.array([[10, 0, 0], [20, 30, 40]]),
        },
        {
            "inputs": np.array([[6, 0, 0], [0, 0, 0]]),
            "targets": np.array([[50, 60, 0], [0, 0, 0]]),
        },
    ]
    expected_segmentations = [
        {
            "inputs": np.array([[1, 1, 1], [1, 1, 0]]),
            "targets": np.array([[1, 0, 0], [1, 1, 1]]),
        },
        {
            "inputs": np.array([[1, 0, 0], [0, 0, 0]]),
            "targets": np.array([[1, 1, 0], [0, 0, 0]]),
        },
    ]
    expected_positions = [
        {
            "inputs": np.array([[0, 1, 2], [0, 1, 0]]),
            "targets": np.array([[0, 0, 0], [0, 1, 2]]),
        },
        {
            "inputs": np.array([[0, 0, 0], [0, 0, 0]]),
            "targets": np.array([[0, 1, 0], [0, 0, 0]]),
        },
    ]
    expected_indexes = [1, 2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )

  def test_pack_sequences_different_data_types(self):
    input_dataset = [
        {
            "inputs": [1, 2, 3],
            "targets": [10.0],
        },
        {
            "inputs": [4, 5],
            "targets": [20.0, 30.0, 40.0],
        },
        {
            "inputs": [6],
            "targets": [50.0, 60.0],
        },
    ]

    length_struct = {"inputs": 3, "targets": 3}
    batch_size = 3

    expected_values = [{
        "inputs": np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]]),
        "targets": np.array(
            [[10.0, 0.0, 0.0], [20.0, 30.0, 40.0], [50.0, 60.0, 0.0]]
        ),
    }]
    expected_segmentations = [{
        "inputs": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
        "targets": np.array([[1, 0, 0], [1, 1, 1], [1, 1, 0]]),
    }]
    expected_positions = [{
        "inputs": np.array([[0, 1, 2], [0, 1, 0], [0, 0, 0]]),
        "targets": np.array([[0, 0, 0], [0, 1, 2], [0, 1, 0]]),
    }]
    expected_indexes = [2]
    expected_packed_dataset = create_expected_dataset(
        expected_values,
        expected_segmentations,
        expected_positions,
        expected_indexes,
    )

    common_test_body(
        input_dataset, expected_packed_dataset, length_struct, batch_size
    )


if __name__ == "__main__":
  absltest.main()
