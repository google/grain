"""Provides BatchAndPackDataset.

This is a custom tf.data operation transforming a tf.data.Dataset. It fuses
packing examples and batching. The input is a Dataset with single examples
where all features are sequences (or scalars). The first dimension is always the
sequence dimension and is allowed to vary in size (but cannot be longer than the
desired sequence length for the feature). If examples are shorter than the
provided sequence length they the op may pack them together. If no more examples
would fit into the batch the remaining space in the patch is filled with
padding and a new batch is started.

For each feature in the input dataset the output dataset will contain a tuple
of 3 elements:
1. The packed values.
2. The segment IDs (starting at 1 for each packed example).
3. The positions of each value within the original example (starting at 0).

All features will be padded with 0s to [batch_size, sequence length]. Any
remaining dimensions must be the same between elements of the dataset.

For examples see the test cases.
"""
from typing import Any

from grain._src.tensorflow.ops import gen_batch_and_pack_op
import tensorflow as tf


def _check_is_tensor_spec(component_spec):
  if not isinstance(component_spec, tf.TensorSpec):
    raise TypeError(
        "`BatchAndPackDataset` is only supported for datasets "
        "that produce tensor elements but the input dataset "
        "produces elements of unsupported type "
        f"{component_spec.value_type}."
    )


class BatchAndPackDataset(tf.data.Dataset):
  """A `Dataset` that batches and packes continuous elements from its input."""

  def __init__(
      self,
      input_dataset: tf.data.Dataset,
      *,
      batch_size: int,
      sequence_lengths: Any,
      parallel_copy: bool = True,
  ):
    self._name = None
    self._input_dataset = input_dataset
    self._batch_size = tf.convert_to_tensor(
        batch_size, dtype=tf.int64, name="batch_size"
    )
    tf.nest.map_structure(_check_is_tensor_spec, input_dataset.element_spec)

    try:
      tf.nest.assert_same_structure(
          input_dataset.element_spec, sequence_lengths
      )
    except ValueError as e:
      raise ValueError(
          "Input dataset and sequence length must have the same structure. You "
          "must provide a sequence length for each feature. In the above "
          "error the first structure is the input dataset and the second "
          "structure is the provided sequence length."
      ) from e

    def _output_tensor_spec(ts: tf.TensorSpec, seq_len: int):
      # The first dimension must be the sequence dimension that gets packed to
      # the desired sequence length. We prepend a dimension for the batch size.
      # We treat scalars as vectors of size 1.
      shape = (batch_size, seq_len) + (ts.shape[1:] if len(ts.shape) else ())
      # We would prefer to use an `OrderedDict` here but `tf.nest.flatten` will
      # ignore the order of the dict.
      return (
          tf.TensorSpec(shape=shape, dtype=ts.dtype),  # Values.
          tf.TensorSpec((batch_size, seq_len), tf.int64),  # Segment IDs.
          tf.TensorSpec((batch_size, seq_len), tf.int64),
      )  # Positions.

    flat_sequence_lengths = tf.nest.flatten(sequence_lengths)
    flat_structure = [
        _output_tensor_spec(ts, seq_len)
        for ts, seq_len in zip(
            tf.nest.flatten(input_dataset.element_spec), flat_sequence_lengths
        )
    ]
    self._structure = tf.nest.pack_sequence_as(
        input_dataset.element_spec, flat_structure
    )

    variant_tensor = gen_batch_and_pack_op.batch_and_pack_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        batch_size=self._batch_size,
        sequence_lengths=flat_sequence_lengths,
        parallel_copy=parallel_copy,
        **self._common_args,
    )
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure

  # Needed by interface.
  def _inputs(self):
    return [self._input_dataset]
