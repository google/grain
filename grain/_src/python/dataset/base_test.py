# Copyright 2024 Google LLC
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
"""Tests for base.py."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python import data_sources
from grain._src.python.dataset import base


class RandomAccessDataSourceTest(parameterized.TestCase):

  @parameterized.parameters(
      data_sources.ArrayRecordDataSource,
      data_sources.RangeDataSource,
      data_sources.InMemoryDataSource,
  )
  def test_protocol(self, source_cls):
    self.assertIsInstance(source_cls, base.RandomAccessDataSource)


if __name__ == "__main__":
  absltest.main()
