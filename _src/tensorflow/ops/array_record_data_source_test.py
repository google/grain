# Copyright 2022 Google LLC
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
"""Tests for array_record_data_source."""
import pathlib
import random

from absl import flags
from grain._src.tensorflow.ops import array_record_data_source
import tensorflow as tf

FLAGS = flags.FLAGS

TfArrayRecordDataSource = array_record_data_source.TfArrayRecordDataSource


def _get_value(serialized_example):
  example = tf.train.Example.FromString(serialized_example)
  return example.features.feature["value"].bytes_list.value[0].decode("utf-8")


class TfArrayRecordTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir)

  def test_len(self):
    ar = TfArrayRecordDataSource([
        self.testdata_dir / "alphabet.array_record-00000-of-00002[0:9]",
        self.testdata_dir / "alphabet.array_record-00001-of-00002[0:6]"
    ])
    self.assertLen(ar, 15)

  def test_len_from_shard_spec(self):
    ar = TfArrayRecordDataSource(self.testdata_dir / "alphabet.array_record@2")
    self.assertLen(ar, 26)

  def test_len_from_shard_pattern(self):
    ar = TfArrayRecordDataSource(self.testdata_dir /
                                 "alphabet.array_record-?????-of-?????")
    self.assertLen(ar, 26)

  def test_len_from_list_of_files(self):
    ar = TfArrayRecordDataSource([
        self.testdata_dir / "alphabet.array_record-00000-of-00002",
        self.testdata_dir / "alphabet.array_record-00001-of-00002"
    ])
    self.assertLen(ar, 26)

  def test_getitem_sequential(self):
    ar = TfArrayRecordDataSource(self.testdata_dir / "alphabet.array_record@2")
    for i in range(26):
      record = ar[i]
      actual_value = _get_value(record.numpy())
      self.assertEqual(actual_value, chr(97 + i))

  def test_getitem_random_order(self):
    ar = TfArrayRecordDataSource(self.testdata_dir / "alphabet.array_record@2")
    expected_values = [(i, chr(97 + i)) for i in range(26)]
    random.shuffle(expected_values)
    for i, expected_value in expected_values:
      record = ar[i]
      actual_value = _get_value(record.numpy())
      self.assertEqual(actual_value, expected_value)

  def test_getitem_from_read_instructions(self):
    ar = TfArrayRecordDataSource([
        self.testdata_dir / "alphabet.array_record-00000-of-00002[2:8]",
        self.testdata_dir / "alphabet.array_record-00001-of-00002[1:7]"
    ])
    indices = list(range(2, 8)) + list(range(14, 20))
    expected_values = enumerate([chr(97 + i) for i in indices])
    for i, expected_value in expected_values:
      record = ar[i]
      actual_value = _get_value(record.numpy())
      self.assertEqual(actual_value, expected_value)


if __name__ == "__main__":
  tf.test.main()
