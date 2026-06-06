# Copyright 2026 Google LLC
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
"""Tests for grain metrics selection and wrapping logic."""

from unittest import mock
from absl.testing import absltest

from grain._src.core import monitoring
_HAS_FAST_MONITORING = False


class MonitoringTest(absltest.TestCase):

  def test_record_bytes_read_and_latency_delegates_to_fast_monitoring(self):
    if not _HAS_FAST_MONITORING:
      self.skipTest("fast_monitoring C++ extension is not available")

    with mock.patch.object(
        fast_monitoring, "record_bytes_read_and_latency", autospec=True
    ) as mock_record:
      monitoring.record_bytes_read_and_latency("source_foo", 100, 500, 1)
      mock_record.assert_called_once_with("source_foo", 100, 500, 1)

  def test_record_bytes_read_and_latency_fallback(self):
    with (
        mock.patch.object(monitoring, "fast_monitoring", None),
        mock.patch.object(
            monitoring._impl, "record_bytes_read_and_latency", autospec=True
        ) as mock_fallback,
    ):
      monitoring.record_bytes_read_and_latency("source_bar", 1000, 8000, 2)
      mock_fallback.assert_called_once_with("source_bar", 1000, 8000, 2)


if __name__ == "__main__":
  absltest.main()
