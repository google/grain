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
"""Python API for merging tf.data.Dataset according to a feature."""
from typing import Sequence

from grain._src.tensorflow.ops import gen_merge_by_min_value_op
import tensorflow as tf


class MergeByMinValueDataset(tf.data.Dataset):
  """A `Dataset` that merges multiple datasets according to a feature value.

  This op is meant for merging (transformed) index datasets produced by Grain.

  All datasets must have the same structure contain dictionieries.
  The dictionaries must contain a feature "merge_field" that contains
  monotonically increasing int64 scalars (aka indices).

  The merged dataset will be a merge sort of all datasets according to the
  merge_field.
  """

  def __init__(
      self,
      datasets: Sequence[tf.data.Dataset],
      *,
      merge_field: str,
  ):
    self._name = None
    self._datasets = datasets
    specs = [ds.element_spec for ds in datasets]
    for i, spec in enumerate(specs):
      if not isinstance(spec, dict) or merge_field not in spec:
        raise ValueError(
            "Elements in input datasets must be dictionaries containing "
            "{merge_field} but got {spec} for input dataset {i}."
        )
      if not spec[merge_field].shape or spec[merge_field].dtype != tf.int64:
        raise ValueError(
            "The merge field must be a scalar of dtype int64 but "
            f"got {spec[merge_field]} for input dataset {i}."
        )
      if i > 0:
        tf.nest.assert_same_structure(specs[0], spec)

    spec = {k: 0 if k == merge_field else v for k, v in specs[0].items()}
    component_index = -1
    for i, v in enumerate(tf.nest.flatten(spec)):
      if v == 0:
        component_index = i
    assert component_index >= 0

    variant_tensor = gen_merge_by_min_value_op.merge_by_min_value_dataset(
        input_datasets=[ds._variant_tensor for ds in datasets],  # pylint: disable=protected-access
        component_index=component_index,
        **self._common_args,
    )
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._datasets[0].element_spec

  def _inputs(self):
    return self._datasets
