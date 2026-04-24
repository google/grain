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
import os
from unittest import mock

from absl.testing import absltest
from grain._src.core import monitoring

# pylint: disable=g-import-not-at-top

try:
  import prometheus_client
except ImportError:
  prometheus_client = None
# pylint: enable=g-import-not-at-top


class MonitoringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Use mock.patch to reset internal state without production hooks.
    # This is standard Google practice for keeping production code clean.
    self.enter_context(mock.patch.object(monitoring, '_initialized', False))
    self.enter_context(mock.patch.object(monitoring, '_prometheus_metrics', {}))

    if prometheus_client:
      # Clear registry for hermetic tests.
      registry = prometheus_client.REGISTRY
      # pylint: disable=protected-access
      if hasattr(registry, '_collector_to_names'):
        for collector in list(registry._collector_to_names):
          registry.unregister(collector)
      # pylint: enable=protected-access

    if 'DISABLE_PYGRAIN_TELEMETRY' in os.environ:
      del os.environ['DISABLE_PYGRAIN_TELEMETRY']

  def test_initialize_prometheus_server_called_once(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')
    with mock.patch.object(
        prometheus_client, 'start_http_server', autospec=True
    ) as mock_start_http_server:
      with mock.patch.object(monitoring, '_USE_PROMETHEUS', True):
        monitoring.initialize(8000)
        mock_start_http_server.assert_called_once_with(8000)
        monitoring.initialize(8000)
        # Still once, because _initialized is True in this context.
        mock_start_http_server.assert_called_once_with(8000)

  def test_kill_switch_prevents_initialization(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')
    os.environ['DISABLE_PYGRAIN_TELEMETRY'] = 'true'
    with mock.patch.object(
        prometheus_client, 'start_http_server', autospec=True
    ) as mock_start_http_server:
      with mock.patch.object(monitoring, '_USE_PROMETHEUS', True):
        monitoring.initialize(8000)
        mock_start_http_server.assert_not_called()

  def test_counter_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        monitoring, '_PROMETHEUS_ALLOWED_METRICS', {'/grain/test/counter'}
    ):
      c = monitoring.Counter(
          '/grain/test/counter',
          metadata=monitoring.Metadata(description='My counter'),
      )
      c.Increment()
      c.IncrementBy(5)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_counter_total'
          ),
          6,
      )

  def test_counter_with_labels_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/counter_labels'},
    ):
      c = monitoring.Counter(
          '/grain/test/counter_labels',
          metadata=monitoring.Metadata(description='Labeled counter'),
          fields=[('label1', str), ('label2', int)],
      )
      c.Increment('a', 1)
      c.IncrementBy(2, 'b', 2)
      c.Increment('a', 1)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_counter_labels_total', {'label1': 'a', 'label2': '1'}
          ),
          2,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_counter_labels_total', {'label1': 'b', 'label2': '2'}
          ),
          2,
      )

  def test_gauge_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        monitoring, '_PROMETHEUS_ALLOWED_METRICS', {'/grain/test/gauge'}
    ):
      g = monitoring.Metric(
          '/grain/test/gauge',
          metadata=monitoring.Metadata(description='My gauge'),
      )
      g.Set(5)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_gauge'
          ),
          5,
      )

  def test_histogram_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        monitoring, '_USE_PROMETHEUS', True
    ), mock.patch.object(
        monitoring, '_PROMETHEUS_ALLOWED_METRICS', {'/grain/test/histogram'}
    ):
      h = monitoring.EventMetric(
          '/grain/test/histogram',
          metadata=monitoring.Metadata(description='My histogram'),
      )
      h.Record(10)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_histogram_count'
          ),
          1,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'grain_test_histogram_sum'
          ),
          10,
      )

  def test_ignore_non_allowed_metrics_prometheus(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(monitoring, '_USE_PROMETHEUS', True):
      # Ensure allowlist is intact and does NOT include the test metric
      c = monitoring.Counter(
          '/not_allowed/counter', metadata=monitoring.Metadata(description='d')
      )
      c.Increment()
      self.assertIsNone(
          prometheus_client.REGISTRY.get_sample_value(  # pytype: disable=attribute-error
              'not_allowed_counter_total'
          )
      )


if __name__ == '__main__':
  absltest.main()
