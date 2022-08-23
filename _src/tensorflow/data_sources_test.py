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
"""Unit tests for the data_sources module."""
from unittest import mock

from grain._src.tensorflow import data_sources
import tensorflow as tf
import tensorflow_datasets as tfds

FileInstruction = tfds.core.utils.shard_utils.FileInstruction


class DatasourcesTest(tf.test.TestCase):
  """Tests for the data_sources module."""

  def test_tfds_data_source_tfrecord(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.TFRECORD
    with self.assertRaisesRegex(
        NotImplementedError,
        "No random access data source for file format FileFormat.TFRECORD"):
      data_sources.TfdsDataSource(tfds_info, split="train")

  def test_tfds_data_source_array_record(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.splits["splits"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 7),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(data_sources,
                           "TfArrayRecordDataSource") as underlying_source_mock:
      data_sources.TfdsDataSource(tfds_info, split="train")
      underlying_source_mock.assert_called_once_with([
          "my_file-000-of-003[0:12]", "my_file-001-of-003[2:9]",
          "my_file-002-of-003[0:4]"
      ])


if __name__ == "__main__":
  tf.test.main()
