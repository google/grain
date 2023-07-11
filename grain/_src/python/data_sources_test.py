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
"""Tests for data sources."""

import dataclasses
import pathlib
import pickle
import random
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python import data_sources
import tensorflow_datasets as tfds


FLAGS = flags.FLAGS
FileInstruction = tfds.core.utils.shard_utils.FileInstruction


@dataclasses.dataclass
class DummyFileInstruction:
  filename: str
  skip: int
  take: int
  examples_in_shard: int


class DataSourceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(FLAGS.test_srcdir)


class RangeDataSourceTest(DataSourceTest):

  @parameterized.parameters([
      (0, 10, 2),  # Positive step
      (10, 0, -2),  # Negative step
      (0, 0, 1),  # Empty Range
  ])
  def test_range_data_source(self, start, stop, step):
    expected_output = list(range(start, stop, step))

    range_ds = data_sources.RangeDataSource(start, stop, step)
    actual_output = [range_ds[i] for i in range(len(range_ds))]

    self.assertEqual(expected_output, actual_output)


class ArrayRecordDataSourceTest(DataSourceTest):

  def test_array_record_data_implements_random_access(self):
    assert issubclass(
        data_sources.ArrayRecordDataSource, data_sources.RandomAccessDataSource
    )

  def test_array_record_source_empty_sequence(self):
    with self.assertRaises(ValueError):
      data_sources.ArrayRecordDataSource([])


class BagDataSourceTest(DataSourceTest):

  def test_bag_data_source_single_file_len(self):
    bag_data_path = self.testdata_dir / "digits-00000-of-00002.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    self.assertLen(bag_ds, 5)

  def test_bag_data_source_sharded_files_len(self):
    bag_data_path = self.testdata_dir / "digits@2.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    self.assertLen(bag_ds, 10)

  def test_bag_data_source_single_file_sequential_get(self):
    bag_data_path = self.testdata_dir / "digits-00000-of-00002.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    expected_data = [b"0", b"1", b"2", b"3", b"4"]
    actual_data = [bag_ds[i] for i in range(5)]
    self.assertEqual(expected_data, actual_data)

  def test_bag_data_source_sharded_files_sequential_get(self):
    bag_data_path = self.testdata_dir / "digits@2.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    expected_data = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
    actual_data = [bag_ds[i] for i in range(10)]
    self.assertEqual(expected_data, actual_data)

  def test_bag_data_source_single_file_reverse_sequential_get(self):
    bag_data_path = self.testdata_dir / "digits-00000-of-00002.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    indices_to_read = [0, 1, 2, 3, 4]
    expected_data = [b"0", b"1", b"2", b"3", b"4"]
    indices_to_read.reverse()
    expected_data.reverse()
    actual_data = [bag_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_bag_data_source_sharded_files_reverse_sequential_get(self):
    bag_data_path = self.testdata_dir / "digits@2.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    indices_to_read = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_data = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
    indices_to_read.reverse()
    expected_data.reverse()
    actual_data = [bag_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_bag_data_source_single_file_random_get(self):
    bag_data_path = self.testdata_dir / "digits-00000-of-00002.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    indices_to_read = [0, 1, 2, 3, 4]
    random.shuffle(indices_to_read)
    data = [b"0", b"1", b"2", b"3", b"4"]
    expected_data = [data[idx] for idx in indices_to_read]
    actual_data = [bag_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_bag_data_source_sharded_files_random_get(self):
    bag_data_path = self.testdata_dir / "digits@2.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    indices_to_read = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(indices_to_read)
    data = [b"0", b"1", b"2", b"3", b"4", b"5", b"6", b"7", b"8", b"9"]
    expected_data = [data[idx] for idx in indices_to_read]
    actual_data = [bag_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_pickle(self):
    bag_data_path = self.testdata_dir / "digits@2.bagz"
    bag_ds = data_sources.BagDataSource(path=bag_data_path)
    serialized_bag_ds = pickle.dumps(bag_ds)
    bag_ds = pickle.loads(serialized_bag_ds)
    self.assertLen(bag_ds, 10)
    # Underlying reader is not pickled.
    self.assertIsNone(bag_ds._reader)
    # Reading still works.
    expected_data = [b"0", b"1", b"2", b"3", b"4"]
    actual_data = [bag_ds[i] for i in range(5)]
    self.assertEqual(expected_data, actual_data)


class TfdsDataSourceTest(DataSourceTest):
  """Tests for TfdsDataSource."""

  def test_tfrecord_file_format_raises_error(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.TFRECORD
    with self.assertRaisesRegex(
        NotImplementedError,
        "No random access data source for file format FileFormat.TFRECORD",
    ):
      with data_sources.TfdsDataSource(tfds_info, split="train"):
        pass

  def test_array_record_file_format_delegates_to_array_record_data_source(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.splits["train"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 11),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(
        data_sources, "ArrayRecordDataSource"
    ) as underlying_source_mock:
      with data_sources.TfdsDataSource(tfds_info, split="train"):
        underlying_source_mock.assert_called_once_with([
            FileInstruction("my_file-000-of-003", 0, 12, 12),
            FileInstruction("my_file-001-of-003", 2, 9, 11),
            FileInstruction("my_file-002-of-003", 0, 4, 4),
        ])

  def test_repr_returns_meaningful_string_without_decoders(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.data_dir = "/path/to/data/dir"
    tfds_info.splits["train"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 11),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(data_sources, "ArrayRecordDataSource"):
      with data_sources.TfdsDataSource(tfds_info, split="train") as source:
        self.assertEqual(
            repr(source),
            (
                "TfdsDataSource(builder_directory='/path/to/data/dir',"
                " split='train', decoders=None)"
            ),
        )

  def test_repr_returns_meaningful_string_with_decoders(self):
    tfds_info = mock.create_autospec(tfds.core.DatasetInfo)
    tfds_info.file_format = tfds.core.file_adapters.FileFormat.ARRAY_RECORD
    tfds_info.data_dir = "/path/to/data/dir"
    tfds_info.splits["train"].file_instructions = [
        FileInstruction("my_file-000-of-003", 0, 12, 12),
        FileInstruction("my_file-001-of-003", 2, 9, 11),
        FileInstruction("my_file-002-of-003", 0, 4, 4),
    ]
    with mock.patch.object(data_sources, "ArrayRecordDataSource"):
      with data_sources.TfdsDataSource(
          tfds_info,
          split="train",
          decoders={"my_feature": tfds.decode.SkipDecoding()},
      ) as source:
        self.assertEqual(
            repr(source),
            (
                "TfdsDataSource(builder_directory='/path/to/data/dir',"
                " split='train', decoders={'my_feature': <class"
                " 'tensorflow_datasets.core.decode.base.SkipDecoding'>})"
            ),
        )


class SSTableDataSourceTest(DataSourceTest):

  def test_len(self):
    with data_sources.SSTableDataSource(
        self.testdata_dir / "digits@2.sst",
        self.testdata_dir / "digits_keys@1.sst",
    ) as ss:
      self.assertLen(ss, 5)

  def test_list_of_files(self):
    with data_sources.SSTableDataSource(
        [
            self.testdata_dir / "digits-00000-of-00002.sst",
            self.testdata_dir / "digits-00001-of-00002.sst",
        ],
        self.testdata_dir / "digits_keys@1.sst",
    ) as ss:
      self.assertLen(ss, 5)

  def test_data_source_reverse_order(self):
    sstable_ds = data_sources.SSTableDataSource(
        self.testdata_dir / "digits@2.sst",
        self.testdata_dir / "digits_keys@1.sst",
    )
    indices_to_read = [4, 3, 2, 1, 0]
    expected_data = [
        b"record_4",
        b"record_3",
        b"record_2",
        b"record_1",
        b"record_0",
    ]
    actual_data = [sstable_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_random_order(self):
    sstable_ds = data_sources.SSTableDataSource(
        self.testdata_dir / "digits@2.sst",
        self.testdata_dir / "digits_keys@1.sst",
    )
    indices_to_read = [2, 0, 4, 1, 3]
    expected_data = [
        b"record_2",
        b"record_0",
        b"record_4",
        b"record_1",
        b"record_3",
    ]
    actual_data = [sstable_ds[i] for i in indices_to_read]
    self.assertEqual(expected_data, actual_data)

  def test_malformed_key_file(self):
    with self.assertRaisesRegex(
        ValueError,
        "The length of the SSTable does not match the number of keys.",
    ):
      with data_sources.SSTableDataSource(
          self.testdata_dir / "digits@2.sst",
          self.testdata_dir / "digits_malformed_keys@1.sst",
      ):
        pass

  def test_duplicated_keys_file(self):
    with self.assertRaisesRegex(
        ValueError, "The SSTableDataSource does not support duplicated keys."
    ):
      with data_sources.SSTableDataSource(
          self.testdata_dir / "digits@2.sst",
          self.testdata_dir / "digits_duplicated_keys@1.sst",
      ):
        pass


if __name__ == "__main__":
  absltest.main()
