"""Test for BatchAndPackDataset."""
from typing import Mapping, Sequence

from absl.testing import parameterized
from grain._src.tensorflow.ops import batch_and_pack
import numpy as np
import tensorflow as tf

Dataset = Sequence[Mapping[str, np.ndarray]]


def flatten_dict(ds):
  """Litter helper function to convert elements back to flat dictionaries."""

  def map_fn(features):
    result = {}
    for k, v in features.items():
      result[k] = v[0]
      result[f"{k}_segment_ids"] = v[1]
      result[f"{k}_positions"] = v[2]
    return result

  return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class BatchAndPackDatasetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the BatchAndPack dataset."""

  def assertDatasetsEqual(
      self, actual_dataset: Dataset, expected_dataset: Dataset
  ):
    self.assertLen(actual_dataset, len(expected_dataset))
    for row in range(len(expected_dataset)):
      features = expected_dataset[row].keys()
      self.assertSetEqual(set(actual_dataset[row].keys()), set(features))
      for k in features:
        self.assertAllEqual(
            actual_dataset[row][k],
            expected_dataset[row][k],
            msg=f"Feature '{k} in row {row} doesn't match.",
        )

  @parameterized.parameters([False, True])
  def test_pack_sequences_length_3(self, parallel_copy: bool):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=3,
        sequence_lengths={"inputs": 3, "targets": 3},
        parallel_copy=parallel_copy,
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3], [4, 5, 0], [6, 0, 0]],
        "inputs_segment_ids": [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        "inputs_positions": [[0, 1, 2], [0, 1, 0], [0, 0, 0]],
        "targets": [[10, 0, 0], [20, 30, 40], [50, 60, 0]],
        "targets_segment_ids": [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
        "targets_positions": [[0, 0, 0], [0, 1, 2], [0, 1, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_length_4(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=2,
        sequence_lengths={"inputs": 4, "targets": 4},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3, 6], [4, 5, 0, 0]],
        "inputs_segment_ids": [[1, 1, 1, 2], [1, 1, 0, 0]],
        "inputs_positions": [[0, 1, 2, 0], [0, 1, 0, 0]],
        "targets": [[10, 50, 60, 0], [20, 30, 40, 0]],
        "targets_segment_ids": [[1, 2, 2, 0], [1, 1, 1, 0]],
        "targets_positions": [[0, 0, 1, 0], [0, 1, 2, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_length_5(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=2,
        sequence_lengths={"inputs": 5, "targets": 5},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3, 4, 5], [6, 0, 0, 0, 0]],
        "inputs_segment_ids": [[1, 1, 1, 2, 2], [1, 0, 0, 0, 0]],
        "inputs_positions": [[0, 1, 2, 0, 1], [0, 0, 0, 0, 0]],
        "targets": [[10, 20, 30, 40, 0], [50, 60, 0, 0, 0]],
        "targets_segment_ids": [[1, 2, 2, 2, 0], [1, 1, 0, 0, 0]],
        "targets_positions": [[0, 0, 1, 2, 0], [0, 1, 0, 0, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_length_6(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=1,
        sequence_lengths={"inputs": 6, "targets": 6},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3, 4, 5, 6]],
        "inputs_segment_ids": [[1, 1, 1, 2, 2, 3]],
        "inputs_positions": [[0, 1, 2, 0, 1, 0]],
        "targets": [[10, 20, 30, 40, 50, 60]],
        "targets_segment_ids": [[1, 2, 2, 2, 3, 3]],
        "targets_positions": [[0, 0, 1, 2, 0, 1]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_length_7(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=1,
        sequence_lengths={"inputs": 7, "targets": 7},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3, 4, 5, 6, 0]],
        "inputs_segment_ids": [[1, 1, 1, 2, 2, 3, 0]],
        "inputs_positions": [[0, 1, 2, 0, 1, 0, 0]],
        "targets": [[10, 20, 30, 40, 50, 60, 0]],
        "targets_segment_ids": [[1, 2, 2, 2, 3, 3, 0]],
        "targets_positions": [[0, 0, 1, 2, 0, 1, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_scalar(self):
    """Scalars should be treated the same as vectors of size 1."""
    input_dataset = tf.data.experimental.from_list([
        {
            "index": 0,
            "inputs": [1, 2, 3],
            "targets": [10],
        },
        {
            "index": 1,
            "inputs": [4, 5],
            "targets": [20, 30, 40],
        },
        {
            "index": 2,
            "inputs": [6],
            "targets": [50, 60],
        },
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=1,
        sequence_lengths={"index": 7, "inputs": 7, "targets": 7},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "index": [[0, 1, 2, 0, 0, 0, 0]],
        "index_segment_ids": [[1, 2, 3, 0, 0, 0, 0]],
        "index_positions": [[0, 0, 0, 0, 0, 0, 0]],
        "inputs": [[1, 2, 3, 4, 5, 6, 0]],
        "inputs_segment_ids": [[1, 1, 1, 2, 2, 3, 0]],
        "inputs_positions": [[0, 1, 2, 0, 1, 0, 0]],
        "targets": [[10, 20, 30, 40, 50, 60, 0]],
        "targets_segment_ids": [[1, 2, 2, 2, 3, 3, 0]],
        "targets_positions": [[0, 0, 1, 2, 0, 1, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_different_lengths(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=3,
        sequence_lengths={"inputs": 3, "targets": 4},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3], [4, 5, 0], [6, 0, 0]],
        "inputs_segment_ids": [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        "inputs_positions": [[0, 1, 2], [0, 1, 0], [0, 0, 0]],
        "targets": [[10, 0, 0, 0], [20, 30, 40, 0], [50, 60, 0, 0]],
        "targets_segment_ids": [[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]],
        "targets_positions": [[0, 0, 0, 0], [0, 1, 2, 0], [0, 1, 0, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_two_dimensional_features(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=2,
        sequence_lengths={
            "input_tokens": 5,
            "input_vectors": 3,
            "targets": 5,
        },
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "input_tokens": [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]],
        "input_tokens_segment_ids": [[1, 1, 1, 0, 0], [1, 1, 2, 0, 0]],
        "input_tokens_positions": [[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]],
        "input_vectors": [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[3, 4, 5], [4, 5, 6], [5, 6, 7]],
        ],
        "input_vectors_segment_ids": [[1, 1, 1], [1, 1, 2]],
        "input_vectors_positions": [[0, 1, 2], [0, 1, 0]],
        "targets": [[10, 0, 0, 0, 0], [20, 30, 40, 50, 60]],
        "targets_segment_ids": [[1, 0, 0, 0, 0], [1, 1, 1, 2, 2]],
        "targets_positions": [[0, 0, 0, 0, 0], [0, 1, 2, 0, 1]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_into_multiple_batches(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=2,
        sequence_lengths={"inputs": 3, "targets": 3},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [
        {
            "inputs": [[1, 2, 3], [4, 5, 0]],
            "inputs_segment_ids": [[1, 1, 1], [1, 1, 0]],
            "inputs_positions": [[0, 1, 2], [0, 1, 0]],
            "targets": [[10, 0, 0], [20, 30, 40]],
            "targets_segment_ids": [[1, 0, 0], [1, 1, 1]],
            "targets_positions": [[0, 0, 0], [0, 1, 2]],
        },
        {
            "inputs": [[6, 0, 0], [0, 0, 0]],
            "inputs_segment_ids": [[1, 0, 0], [0, 0, 0]],
            "inputs_positions": [[0, 0, 0], [0, 0, 0]],
            "targets": [[50, 60, 0], [0, 0, 0]],
            "targets_segment_ids": [[1, 1, 0], [0, 0, 0]],
            "targets_positions": [[0, 1, 0], [0, 0, 0]],
        },
    ]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_pack_sequences_different_data_types(self):
    input_dataset = tf.data.experimental.from_list([
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
    ])
    ds = batch_and_pack.BatchAndPackDataset(
        input_dataset,
        batch_size=3,
        sequence_lengths={"inputs": 3, "targets": 3},
    )
    ds = flatten_dict(ds)
    actual_dataset = list(ds.as_numpy_iterator())

    expected_dataset = [{
        "inputs": [[1, 2, 3], [4, 5, 0], [6, 0, 0]],
        "inputs_segment_ids": [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        "inputs_positions": [[0, 1, 2], [0, 1, 0], [0, 0, 0]],
        "targets": [[10, 0, 0], [20, 30, 40], [50, 60, 0]],
        "targets_segment_ids": [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
        "targets_positions": [[0, 0, 0], [0, 1, 2], [0, 1, 0]],
    }]
    self.assertDatasetsEqual(actual_dataset, expected_dataset)

  def test_invalid_dataset_spec(self):
    input_dataset = tf.data.Dataset.range(1).map(
        lambda x: tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
    )
    with self.assertRaisesRegex(
        TypeError, "^`BatchAndPackDataset`.*RaggedTensor"
    ):
      batch_and_pack.BatchAndPackDataset(
          input_dataset, batch_size=2, sequence_lengths={}
      )

  def test_invalid_dataset_structure_mismatch(self):
    input_dataset = tf.data.experimental.from_list([{
        "inputs": [1, 2, 3],
        "targets": [10.0],
    }])
    with self.assertRaisesRegex(
        ValueError,
        "Input dataset and sequence length must have the same structure.",
    ):
      batch_and_pack.BatchAndPackDataset(
          input_dataset,
          batch_size=3,
          sequence_lengths={
              "index": 3,
          },
      )


if __name__ == "__main__":
  tf.test.main()
