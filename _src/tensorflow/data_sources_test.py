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

from absl.testing import parameterized
from grain._src.tensorflow import data_sources
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

FileInstruction = tfds.core.utils.shard_utils.FileInstruction
_EMPTY_IMAGENET_EXAMPLE = """\n\xc0\x05\n\x87\x05\n\x05image\x12\xfd\x04\n\xfa\x04\n\xf7\x04\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01,\x01,\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xfe\x7f\xe8\xa2\x8a\x00\xff\xd9\n#\n\tfile_name\x12\x16\n\x14\n\x12n02165105_115.JPEG\n\x0f\n\x05label\x12\x06\x1a\x04\n\x02\xac\x02"""


class TfdsDataSourceTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for TfdsDataSource."""

  def test_tfds_data_source_tfrecord(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.TFRECORD
    with self.assertRaisesRegex(
        NotImplementedError,
        "No random access data source for file format FileFormat.TFRECORD",
    ):
      data_sources.TfdsDataSource(tfds_info, split="train")

  @parameterized.parameters([False, True])
  def test_tfds_data_source_array_record(self, cache: bool):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.splits["splits"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 11),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(
        data_sources, "TfArrayRecordDataSource"
    ) as underlying_source_mock:
      data_sources.TfdsDataSource(tfds_info, split="train", cache=cache)
      underlying_source_mock.assert_called_once_with(
          [
              "my_file-000-of-003[0:12]",
              "my_file-001-of-003[2:11]",
              "my_file-002-of-003[0:4]",
          ],
          cache=cache,
      )

  def test_repr(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.data_dir = "/path/to/data/dir"
    tfds_info.splits["train"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 11),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(data_sources, "TfArrayRecordDataSource"):
      source = data_sources.TfdsDataSource(
          tfds_info, split="train", decoders=None
      )
      self.assertEqual(
          repr(source),
          (
              "TfdsDataSource(builder_directory='/path/to/data/dir',"
              " split='train', decoders=None)"
          ),
      )
      source = data_sources.TfdsDataSource(
          tfds_info,
          split="train",
          decoders={"my_feature": tfds.decode.SkipDecoding()},
      )
      self.assertEqual(
          repr(source),
          (
              "TfdsDataSource(builder_directory='/path/to/data/dir',"
              " split='train', decoders={'my_feature': <class"
              " 'tensorflow_datasets.core.decode.base.SkipDecoding'>})"
          ),
      )


class TfInMemoryDataSourceTest(tf.test.TestCase):
  """Tests for TfInMemoryDataSource."""

  def test_from_dataset(self):
    ds = tf.data.Dataset.range(10)
    source = data_sources.TfInMemoryDataSource.from_dataset(ds)
    self.assertLen(source, 10)
    self.assertEqual(source[2], 2)
    self.assertAllEqual(source[(3, 7)], (3, 7))

  def test_from_data_frame(self):
    data = {"col1": [0, 1, 2, 3], "col2": pd.Series([2, 3], index=[2, 3])}
    df = pd.DataFrame(data=data, index=[0, 1, 2, 3])
    source = data_sources.TfInMemoryDataSource.from_data_frame(df)
    self.assertLen(source, 4)
    self.assertAllEqual(source[0]["col1"], 0)
    self.assertAllEqual(source[1]["col1"], 1)
    self.assertTrue(tf.math.is_nan(source[0]["col2"]))
    self.assertTrue(tf.math.is_nan(source[1]["col2"]))
    self.assertAllClose(
        source[2],
        {"col1": tf.cast(2, tf.int64), "col2": tf.cast(2, tf.float64)},
    )
    self.assertAllClose(
        source[3],
        {"col1": tf.cast(3, tf.int64), "col2": tf.cast(3, tf.float64)},
    )


if __name__ == "__main__":
  tf.test.main()
