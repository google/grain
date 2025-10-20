# Copyright 2025 Google LLC
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
import pathlib

from absl import flags
from absl.testing import absltest
from grain._src.python.dataset.sources import tfrecord_dataset
import grain.python as pygrain


def setup_module():
  # Set the path to test data when run via pytest.
  # When run via bazel, FLAGS.test_srcdir is set from the
  # BUILD file, see args = ["--test_srcdir=grain/_src/python"]
  # in grain/_src/python/dataset/sources/BUILD
  import grain  # pylint: disable=g-import-not-at-top

  srcdir = pathlib.Path(grain.__file__).parents[0] / "_src" / "python"
  flags.FLAGS["test_srcdir"].parse(str(srcdir))


class TFRecordIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.testdata_dir = pathlib.Path(flags.FLAGS.test_srcdir)
    self.testdata_file_path = self.testdata_dir
    self.testdata_file_path /= "testdata"
    self.testdata_file_path /= "morris_sequence_first_5.tfrecord"
    self.expected_data = [
        b"1",
        b"1 1",
        b"2 1",
        b"1 2 1 1",
        b"1 1 1 2 2 1",
    ]

  def test_nonexistent_tfrecord_file(self):
    dataset = tfrecord_dataset.TFRecordIterDataset(
        str(self.testdata_dir / "non_existent_file.tfrecord")
    )
    with self.assertRaises(FileNotFoundError):
      list(dataset)

  def test_empty_tfrecord_file(self):
    empty_tf_record_file = self.create_tempfile("empty_file.tfrecord")
    dataset = tfrecord_dataset.TFRecordIterDataset(
        empty_tf_record_file.full_path
    )
    self.assertSequenceEqual(list(dataset), [])

  def test_invalid_tfrecord_file(self):
    truncated_length_tf_record_file = self.create_tempfile(
        "truncated_length_file.tfrecord"
    )
    # Directly write the data to the file instead of using the tfrecord writer,
    # and without the length prefix. This will create an invalid tfrecord file.
    with open(truncated_length_tf_record_file, "wb") as f:
      f.write(b"1")

    dataset = tfrecord_dataset.TFRecordIterDataset(
        truncated_length_tf_record_file.full_path
    )
    with self.assertRaises(ValueError):
      list(dataset)

  def test_read_tfrecord_file(self):
    dataset = tfrecord_dataset.TFRecordIterDataset(str(self.testdata_file_path))
    self.assertSequenceEqual(list(dataset), self.expected_data)

  def test_checkpointing(self):
    dataset = tfrecord_dataset.TFRecordIterDataset(str(self.testdata_file_path))
    pygrain.experimental.assert_equal_output_after_checkpoint(dataset)


if __name__ == "__main__":
  absltest.main()
