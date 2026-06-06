# pytype: skip-file
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
import errno
import os
import tempfile
from unittest import mock

from absl.testing import absltest

from grain._src.core import monitoring
from grain._src.core import prometheus_monitoring

# pylint: disable=g-import-not-at-top
try:
  import prometheus_client
except ImportError:
  prometheus_client = None


class MonitoringTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(prometheus_monitoring, '_initialized', False)
    )
    self.enter_context(
        mock.patch.object(prometheus_monitoring, '_prometheus_metrics', {})
    )

    if prometheus_client:
      # Clear registry for hermetic tests.
      registry = prometheus_client.REGISTRY
      collector_to_names = getattr(registry, '_collector_to_names', None)
      if collector_to_names:
        for collector in list(collector_to_names):
          registry.unregister(collector)

  def test_initialize_prometheus_server_called_once(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')
    with mock.patch.object(
        prometheus_client, 'start_http_server', autospec=True
    ) as mock_start_http_server:
      prometheus_monitoring.Initialize(9431)
      mock_start_http_server.assert_called_once_with(9431)
      prometheus_monitoring.Initialize(9431)
      # Still once, because _initialized is True in this context.
      mock_start_http_server.assert_called_once_with(9431)

  def test_counter_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/counter'},
    ):
      c = prometheus_monitoring.Counter(
          '/grain/test/counter',
          metadata=monitoring.Metadata(description='My counter'),
      )
      c.Increment()
      c.IncrementBy(5)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              'grain_test_counter_total'
          ),
          6,
      )

  def test_counter_with_labels_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/counter_labels'},
    ):
      c = prometheus_monitoring.Counter(
          '/grain/test/counter_labels',
          metadata=monitoring.Metadata(description='Labeled counter'),
          fields=[('label1', str), ('label2', int)],
      )
      c.Increment('a', 1)
      c.IncrementBy(2, 'b', 2)
      c.Increment('a', 1)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              'grain_test_counter_labels_total', {'label1': 'a', 'label2': '1'}
          ),
          2,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              'grain_test_counter_labels_total', {'label1': 'b', 'label2': '2'}
          ),
          2,
      )

  def test_gauge_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/gauge'},
    ):
      g = prometheus_monitoring.Metric(
          '/grain/test/gauge',
          metadata=monitoring.Metadata(description='My gauge'),
      )
      g.Set(5)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value('grain_test_gauge'),
          5,
      )

  def test_histogram_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/histogram'},
    ):
      h = prometheus_monitoring.EventMetric(
          '/grain/test/histogram',
          metadata=monitoring.Metadata(description='My histogram'),
      )
      h.Record(10)
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              'grain_test_histogram_count'
          ),
          1,
      )
      self.assertEqual(
          prometheus_client.REGISTRY.get_sample_value(
              'grain_test_histogram_sum'
          ),
          10,
      )

  def test_ignore_non_allowed_metrics_prometheus(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    # Ensure allowlist is intact and does NOT include the test metric
    c = prometheus_monitoring.Counter(
        '/not_allowed/counter', metadata=monitoring.Metadata(description='d')
    )
    c.Increment()
    self.assertIsNone(
        prometheus_client.REGISTRY.get_sample_value('not_allowed_counter_total')
    )

  def test_initialize_multiprocess(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with tempfile.TemporaryDirectory() as tmp_dir:
      with mock.patch.dict(os.environ, {'PROMETHEUS_MULTIPROC_DIR': tmp_dir}):
        with mock.patch.object(
            prometheus_client, 'start_http_server'
        ) as mock_start:
          prometheus_monitoring.Initialize(9431)
          mock_start.assert_called_once()
          args, kwargs = mock_start.call_args
          self.assertEqual(args[0], 9431)
          self.assertIn('registry', kwargs)

  def test_setup_telemetry_main_process(self):
    mock_process = mock.Mock()
    mock_process.name = 'MainProcess'
    with mock.patch.dict(
        os.environ, {'ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY': 'true'}
    ):
      with mock.patch(
          'multiprocessing.current_process', return_value=mock_process
      ):
        with mock.patch.object(
            prometheus_monitoring, 'Initialize'
        ) as mock_init:
          prometheus_monitoring.SetupTelemetry()
          mock_init.assert_called_once_with(port=9431)

  def test_setup_telemetry_worker_process(self):
    mock_process = mock.Mock()
    mock_process.name = 'Worker-1'
    with mock.patch.dict(
        os.environ, {'ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY': 'true'}
    ):
      with mock.patch(
          'multiprocessing.current_process', return_value=mock_process
      ):
        with mock.patch.object(
            prometheus_monitoring, 'Initialize'
        ) as mock_init:
          prometheus_monitoring.SetupTelemetry()
          mock_init.assert_called_once_with(port=0)

  def test_initialize_port_already_in_use(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_client,
        'start_http_server',
        side_effect=OSError(errno.EADDRINUSE, 'Address already in use'),
    ):
      with self.assertLogs(level='INFO') as log:
        prometheus_monitoring.Initialize(9431)
        self.assertTrue(getattr(prometheus_monitoring, '_initialized'))
        self.assertTrue(
            any('Prometheus server already active' in m for m in log.output)
        )

  def test_initialize_other_oserror(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_client,
        'start_http_server',
        side_effect=OSError(errno.EINVAL, 'Some other error'),
    ):
      with self.assertLogs(level='WARNING') as log:
        prometheus_monitoring.Initialize(9431)
        self.assertFalse(getattr(prometheus_monitoring, '_initialized'))
        self.assertTrue(
            any('Failed to start Prometheus server' in m for m in log.output)
        )

  def test_ignore_non_allowed_metrics_prometheus_gauge(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    g = prometheus_monitoring.Metric(
        '/not_allowed/gauge', metadata=monitoring.Metadata(description='d')
    )
    g.Set(5)
    self.assertIsNone(
        prometheus_client.REGISTRY.get_sample_value('not_allowed_gauge')
    )

  def test_ignore_non_allowed_metrics_prometheus_event(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    h = prometheus_monitoring.EventMetric(
        '/not_allowed/histogram',
        metadata=monitoring.Metadata(description='d'),
    )
    h.Record(5)
    self.assertIsNone(
        prometheus_client.REGISTRY.get_sample_value(
            'not_allowed_histogram_count'
        )
    )

  def test_missing_prometheus_metrics(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring, '_PROMETHEUS_ALLOWED_METRICS', {'/grain/test'}
    ):
      with mock.patch.object(
          prometheus_monitoring, '_prom_counter', None
      ), mock.patch.object(
          prometheus_monitoring, '_prom_gauge', None
      ), mock.patch.object(
          prometheus_monitoring, '_prom_histogram', None
      ):
        c = prometheus_monitoring.Counter('/grain/test')
        self.assertIsNone(c._metric)
        m = prometheus_monitoring.Metric('/grain/test')
        self.assertIsNone(m._metric)
        e = prometheus_monitoring.EventMetric('/grain/test')
        self.assertIsNone(e._metric)

  def test_histogram_with_bucketer_prometheus_routing(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    with mock.patch.object(
        prometheus_monitoring,
        '_PROMETHEUS_ALLOWED_METRICS',
        {'/grain/test/histogram_bucketer'},
    ):
      bucketer = monitoring.Bucketer.PowersOf(2.0)
      h = prometheus_monitoring.EventMetric(
          '/grain/test/histogram_bucketer',
          metadata=monitoring.Metadata(
              description='d', units=monitoring.Units.MILLISECONDS
          ),
          bucketer=bucketer,
      )
      prometheus_metric = getattr(h, '_metric', None)
      if prometheus_metric:
        upper_bounds = getattr(prometheus_metric, '_upper_bounds', None)
        if upper_bounds:
          self.assertIn(1.0, upper_bounds)
          self.assertIn(2.0, upper_bounds)
          self.assertIn(4.0, upper_bounds)

  def test_record_bytes_read_and_latency(self):
    if prometheus_client is None:
      self.skipTest('prometheus-client not installed')

    prometheus_monitoring._bytes_read = prometheus_monitoring.Counter(
        '/grain/python/data_sources/bytes_read',
        monitoring.Metadata(
            description=(
                'Number of bytes produced by a data source via random access.'
            )
        ),
        fields=[('source', str)],
    )
    prometheus_monitoring._source_read_time_ns = (
        prometheus_monitoring.EventMetric(
            '/grain/python/dataset/source_read_time_ns',
            metadata=monitoring.Metadata(
                description='Histogram of source read time in nanoseconds.',
                units=monitoring.Units.NANOSECONDS,
            ),
            bucketer=monitoring.Bucketer.PowersOf(4.0),
            fields=[('source', str)],
        )
    )

    prometheus_monitoring.record_bytes_read_and_latency(
        'source_baz', 1000, 8000, 2
    )

    self.assertEqual(
        prometheus_client.REGISTRY.get_sample_value(
            'grain_python_data_sources_bytes_read_total',
            labels={'source': 'source_baz'},
        ),
        1000,
    )
    self.assertEqual(
        prometheus_client.REGISTRY.get_sample_value(
            'grain_python_dataset_source_read_time_ns_sum',
            labels={'source': 'source_baz'},
        ),
        8000,
    )
    self.assertEqual(
        prometheus_client.REGISTRY.get_sample_value(
            'grain_python_dataset_source_read_time_ns_count',
            labels={'source': 'source_baz'},
        ),
        2,
    )


if __name__ == '__main__':
  absltest.main()
