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
from grain._src.tensorflow import data_sources
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasourcesTest(tf.test.TestCase):
  """Tests for the data_sources module."""

  def test_tfds_data_source(self):
    with tfds.testing.mock_data():
      tfds_info = tfds.builder("imagenet2012").info
    # TODO(cameliahanes): Figure out how to test TFDS ArrayRecord datasets.
    with self.assertRaisesRegex(
        NotImplementedError,
        "No random access data source for file format FileFormat.TFRECORD"):
      data_sources.TfdsDataSource(tfds_info, split="train[:1000]")


if __name__ == "__main__":
  tf.test.main()
