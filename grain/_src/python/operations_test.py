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
"""Tests for Operations."""

import collections
import dataclasses
import re

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing
from grain._src.python import record
from grain._src.python.operations import BatchOperation
from grain._src.python.operations import FilterOperation
from grain._src.python.operations import MapOperation
from grain._src.python.operations import RandomMapOperation
import numpy as np


class OperationsTest(parameterized.TestCase):

  def compare_output(self, actual, expected):
    self.assertEqual(len(actual), len(expected))
    for a, b in zip(actual, expected):
      # np testing does not compare nested np arrays element by elements when
      # a dataclass is used, hence converting dataclasses to tuples.
      np.testing.assert_equal(dataclasses.astuple(a), dataclasses.astuple(b))

  def test_map_operation(self):
    map_operation = MapOperation(map_function=lambda data: data + 1)
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=3), 1),
        record.Record(record.RecordMetadata(index=1, record_key=2), 2),
        record.Record(record.RecordMetadata(index=2, record_key=1), 3),
        record.Record(record.RecordMetadata(index=3, record_key=0), 4),
    ])

    expected_output_data = [
        record.Record(record.RecordMetadata(index=0, record_key=None), 2),
        record.Record(record.RecordMetadata(index=1, record_key=None), 3),
        record.Record(record.RecordMetadata(index=2, record_key=None), 4),
        record.Record(record.RecordMetadata(index=3, record_key=None), 5),
    ]
    actual_output_data = list(map_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_random_map_operation(self):
    delta = 0.1
    random_map_operation = RandomMapOperation(
        random_map_function=lambda data, rng: data + rng.uniform(-delta, delta)
    )

    input_rng = np.random.Generator(np.random.Philox(key=0))
    input_record = record.Record(
        record.RecordMetadata(index=0, record_key=1, rng=input_rng), 1
    )
    input_data = iter([input_record])

    actual_output_data = list(random_map_operation(input_data))
    self.assertLen(actual_output_data, 1)
    actual_output_record = actual_output_data[0]
    self.assertIsInstance(actual_output_record, record.Record)
    self.assertEqual(
        actual_output_record.metadata.index, input_record.metadata.index
    )
    self.assertIsNone(actual_output_record.metadata.record_key)
    self.assertEqual(actual_output_record.metadata.rng, input_rng)
    self.assertAlmostEqual(
        actual_output_record.data, input_record.data, delta=delta
    )

  def test_map_operation_with_callable_objects(self):
    class CallableObject:

      def __call__(self, x):
        return x + 1

    map_operation = MapOperation(map_function=CallableObject())
    input_data = iter(
        [record.Record(record.RecordMetadata(index=0, record_key=3), 1)]
    )

    expected_output_data = [
        record.Record(record.RecordMetadata(index=0, record_key=None), 2)
    ]
    actual_output_data = list(map_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_filter_operation(self):
    filter_operation = FilterOperation(
        condition_function=lambda data: data % 2 == 0
    )
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=3), 1),
        record.Record(record.RecordMetadata(index=1, record_key=2), 2),
        record.Record(record.RecordMetadata(index=2, record_key=1), 3),
        record.Record(record.RecordMetadata(index=3, record_key=0), 4),
    ])

    expected_output_data = [
        record.Record(record.RecordMetadata(index=1, record_key=None), 2),
        record.Record(record.RecordMetadata(index=3, record_key=None), 4),
    ]

    actual_output_data = list(filter_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  ############ BATCH TESTS ############

  def test_batch_integer_scalars(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=3), 1),
        record.Record(record.RecordMetadata(index=1, record_key=2), 2),
        record.Record(record.RecordMetadata(index=2, record_key=1), 3),
        record.Record(record.RecordMetadata(index=3, record_key=0), 4),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])
        ),
        record.Record(
            record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])
        ),
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_with_remainder(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), 1),
        record.Record(record.RecordMetadata(index=1, record_key=3), 2),
        record.Record(record.RecordMetadata(index=2, record_key=2), 3),
        record.Record(record.RecordMetadata(index=3, record_key=1), 4),
        record.Record(record.RecordMetadata(index=4, record_key=0), 5),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])
        ),
        record.Record(
            record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])
        ),
        record.Record(
            record.RecordMetadata(index=4, record_key=None), np.asarray([5])
        ),
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_drop_remainder(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), 1),
        record.Record(record.RecordMetadata(index=1, record_key=3), 2),
        record.Record(record.RecordMetadata(index=2, record_key=2), 3),
        record.Record(record.RecordMetadata(index=3, record_key=1), 4),
        record.Record(record.RecordMetadata(index=4, record_key=0), 5),
    ])
    batch_operation = BatchOperation(batch_size=2, drop_remainder=True)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None), np.asarray([1, 2])
        ),
        record.Record(
            record.RecordMetadata(index=3, record_key=None), np.asarray([3, 4])
        ),
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_dict_of_numpy_arrays_no_shared_memory(self):
    input_data = iter([
        record.Record(
            record.RecordMetadata(index=0, record_key=4),
            {"a": np.array([1, 2], dtype=np.int32)},
        ),
        record.Record(
            record.RecordMetadata(index=1, record_key=3),
            {"a": np.array([3, 4], dtype=np.int64)},
        ),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            {"a": np.array([[1, 2], [3, 4]])},
        )
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_tuples(self):
    input_data = iter([
        record.Record(
            record.RecordMetadata(index=0, record_key=4), (1, 10.0, "text")
        ),
        record.Record(
            record.RecordMetadata(index=1, record_key=3),
            (2, 20.0, "text_2"),
        ),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            (
                np.array([1, 2]),
                np.array([10.0, 20.0]),
                np.array(["text", "text_2"]),
            ),
        )
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_named_tuples(self):
    point = collections.namedtuple("point", "x y")
    input_data = iter([
        record.Record(
            record.RecordMetadata(index=0, record_key=4),
            point(x=1, y="str"),
        ),
        record.Record(
            record.RecordMetadata(index=1, record_key=3),
            point(x=2, y="str_2"),
        ),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            point(x=np.array([1, 2]), y=np.array(["str", "str_2"])),
        )
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_lists(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), [1, 2]),
        record.Record(record.RecordMetadata(index=1, record_key=3), [3, 4]),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            np.array([[1, 3], [2, 4]]),
        )
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_mixed_scalar_numbers(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), 1),
        record.Record(record.RecordMetadata(index=1, record_key=3), 2.0),
        record.Record(
            record.RecordMetadata(index=2, record_key=2), np.int32(3)
        ),
        record.Record(
            record.RecordMetadata(index=3, record_key=1), np.int64(4)
        ),
        record.Record(record.RecordMetadata(index=4, record_key=0), float(5.0)),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None), np.array([1, 2])
        ),
        record.Record(
            record.RecordMetadata(index=3, record_key=None), np.array([3, 4])
        ),
        record.Record(
            record.RecordMetadata(index=4, record_key=None), np.array([5])
        ),
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)

  def test_batch_lists_different_length(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), [1, 2]),
        record.Record(record.RecordMetadata(index=1, record_key=3), [3, 4, 5]),
    ])
    batch_operation = BatchOperation(batch_size=2)
    with np.testing.assert_raises_regex(
        TypeError,
        re.escape(
            "Record structures do not match. Record at position 0 has "
            "structure [1, 2], while records at positions [1] "
            "have structures [[3, 4, 5]]."
        ),
    ):
      list(batch_operation(input_data))

  def test_batch_dicts_different_keys(self):
    input_data = iter([
        record.Record(record.RecordMetadata(index=0, record_key=4), {"a": 1}),
        record.Record(record.RecordMetadata(index=1, record_key=3), {"b": 2}),
    ])
    batch_operation = BatchOperation(batch_size=2)
    with np.testing.assert_raises_regex(
        TypeError,
        re.escape(
            "Record structures do not match. Record at position 0 has "
            "structure {'a': 1}, while records at positions [1]"
            " have structures [{'b': 2}]."
        ),
    ):
      list(batch_operation(input_data))

  def test_batch_unsupported_type(self):
    @dataclasses.dataclass
    class UnSupportedType:
      some_value: int

    input_data = iter([
        record.Record(
            record.RecordMetadata(index=0, record_key=4), UnSupportedType(1)
        ),
        record.Record(
            record.RecordMetadata(index=1, record_key=3), UnSupportedType(2)
        ),
    ])
    batch_operation = BatchOperation(batch_size=2)
    expected_output_data = [
        record.Record(
            record.RecordMetadata(index=1, record_key=None),
            np.array([UnSupportedType(1), UnSupportedType(2)]),
        )
    ]
    actual_output_data = list(batch_operation(input_data))
    self.compare_output(actual_output_data, expected_output_data)


if __name__ == "__main__":
  absltest.main()
