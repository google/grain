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
"""Tests for record."""

from grain._src.python import record
import numpy as np
from absl.testing import absltest


class RecordTest(absltest.TestCase):

  def test_RecordMetadata_str(self):
    record_metadata = record.RecordMetadata(
        index=0, record_key=0, rng=np.random.default_rng()
    )
    self.assertEqual(
        str(record_metadata),
        "RecordMetadata(index=0, record_key=0, rng=Generator(PCG64))",
    )

  def test_RecordMetadata_str_none_rng(self):
    record_metadata = record.RecordMetadata(index=0, record_key=0)
    self.assertStartsWith(
        str(record_metadata),
        "RecordMetadata(index=0, record_key=0, rng=None",
    )


if __name__ == "__main__":
  absltest.main()
