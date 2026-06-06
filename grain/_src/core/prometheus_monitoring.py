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
"""Grain metrics Prometheus implementation."""

from __future__ import annotations

import atexit
import errno
import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
from typing import Any, cast

from grain._src.core import monitoring_base

Bucketer = monitoring_base.Bucketer
Metadata = monitoring_base.Metadata
Units = monitoring_base.Units

_DEFAULT_PROMETHEUS_PORT = 9431

# Keep a global reference so the directory is not deleted until the program
# exits.
_prometheus_multiproc_dir = None

try:
  # pylint: disable=g-import-not-at-top
  import prometheus_client  # pytype: disable=import-error
  from prometheus_client import multiprocess  # pytype: disable=import-error

  prometheus_client = cast(Any, prometheus_client)
  multiprocess = cast(Any, multiprocess)

  _prom_counter = prometheus_client.Counter
  _prom_gauge = prometheus_client.Gauge
  _prom_histogram = prometheus_client.Histogram
except (ImportError, AttributeError):
  prometheus_client = None
  _prom_counter = None
  _prom_gauge = None
  _prom_histogram = None

_initialized = False
_prometheus_metrics = {}  # Maps metric names to their Prometheus objects.
_lock = threading.Lock()

_PROMETHEUS_ALLOWED_METRICS = {
    '/grain/python/dataset/next_duration_ns',
    '/grain/python/dataset/prefetch_buffer_ready_count',
    '/grain/python/data_sources/bytes_read',
    '/grain/python/dataset/source_read_time_ns',
    '/grain/python/data_loader/iterator_get_next',
}


def _IsAllowed(metric_name: str) -> bool:
  """Returns True if the metric is allowed for Prometheus export."""
  return metric_name in _PROMETHEUS_ALLOWED_METRICS


# pylint: disable=invalid-name
def get_monitoring_root() -> None:
  """Returns None as Prometheus does not have a monitoring root."""
  return None


# pylint: enable=invalid-name


_METADATA_TYPES = (Metadata,)


def _ExtractMetadataAndFields(
    metadata_candidate: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, list[tuple[str, Any]], tuple[Any, ...]]:
  """Robustly extracts metadata and fields from positional or keyword args."""
  _metadata = kwargs.pop('metadata', None)
  _fields = kwargs.pop('fields', None)

  all_pos = []
  if metadata_candidate is not None:
    all_pos.append(metadata_candidate)
  all_pos.extend(args)

  remaining = []
  for arg in all_pos:
    if _metadata is None and isinstance(arg, _METADATA_TYPES):
      _metadata = arg
    elif _fields is None and isinstance(arg, (list, tuple)):
      _fields = arg
    else:
      remaining.append(arg)

  return _metadata, (_fields or []), tuple(remaining)


class Counter:
  """Prometheus Counter wrapper."""

  def __init__(self, name, metadata=None, *args, **kwargs):
    _metadata, _fields, _ = _ExtractMetadataAndFields(metadata, args, kwargs)

    if not _IsAllowed(name) or not _prom_counter:
      self._metric = None
      return

    prom_name = name.strip('/').replace('/', '_')
    labelnames = tuple(f[0] for f in _fields)
    description = 'Grain Counter'
    if _metadata and hasattr(_metadata, 'description'):
      description = _metadata.description

    with _lock:
      if prom_name not in _prometheus_metrics:
        _prometheus_metrics[prom_name] = _prom_counter(
            prom_name, description, labelnames=labelnames
        )
      self._metric = _prometheus_metrics[prom_name]

  def Increment(self, *args, **kwargs):
    if not self._metric:
      return
    if not args and not kwargs:
      self._metric.inc()
      return
    try:
      self._metric.labels(*args, **kwargs).inc()
    except ValueError as e:
      logging.warning(
          'Failed to record Prometheus event due to label mismatch: %s.', e
      )

  def IncrementBy(self, value, *args, **kwargs):
    if not self._metric:
      return
    if not args and not kwargs:
      self._metric.inc(value)
      return
    try:
      self._metric.labels(*args, **kwargs).inc(value)
    except ValueError as e:
      logging.warning(
          'Failed to record Prometheus event due to label mismatch: %s.', e
      )

  def Get(self, *args, **kwargs):
    """Returns 0.0 as Prometheus histograms do not support Get."""
    del args, kwargs
    return 0.0

  def ClearAll(self):
    """No-op for Prometheus histograms."""
    pass


class Metric:
  """Prometheus Gauge wrapper."""

  def __init__(self, name, value_type=float, metadata=None, *args, **kwargs):
    del value_type
    _metadata, _fields, _ = _ExtractMetadataAndFields(metadata, args, kwargs)

    if not _IsAllowed(name) or not _prom_gauge:
      self._metric = None
      return

    prom_name = name.strip('/').replace('/', '_')
    labelnames = tuple(f[0] for f in _fields)
    description = 'Grain Gauge'
    if _metadata and hasattr(_metadata, 'description'):
      description = _metadata.description

    with _lock:
      if prom_name not in _prometheus_metrics:
        _prometheus_metrics[prom_name] = _prom_gauge(
            prom_name, description, labelnames=labelnames
        )
      self._metric = _prometheus_metrics[prom_name]

  def Set(self, value, *args, **kwargs):
    if not self._metric:
      return
    if not args and not kwargs:
      self._metric.set(value)
      return
    try:
      self._metric.labels(*args, **kwargs).set(value)
    except ValueError as e:
      logging.warning(
          'Failed to record Prometheus gauge due to label mismatch: %s.', e
      )

  def Get(self, *args, **kwargs):
    """Returns 0.0 as Prometheus histograms do not support Get."""
    del args, kwargs
    return 0.0

  def ClearAll(self):
    """No-op for Prometheus histograms."""
    pass


class EventMetric:
  """Prometheus Histogram wrapper."""

  def __init__(self, name, metadata=None, *args, **kwargs):
    _metadata, _fields, _ = _ExtractMetadataAndFields(metadata, args, kwargs)

    if not _IsAllowed(name) or not _prom_histogram:
      self._metric = None
      return

    prom_name = name.strip('/').replace('/', '_')
    labelnames = tuple(f[0] for f in _fields)
    description = 'Grain Histogram'
    if _metadata and hasattr(_metadata, 'description'):
      description = _metadata.description

    # Support both 'buckets' and 'bucketer' keywords.
    buckets = kwargs.get('buckets')
    bucketer = kwargs.get('bucketer')

    if buckets is None and isinstance(bucketer, Bucketer):
      units = getattr(_metadata, 'units', None)
      buckets = self._GetPrometheusBuckets(bucketer, units=units)

    with _lock:
      if prom_name not in _prometheus_metrics:
        construct_kwargs = {'labelnames': labelnames}
        if buckets:
          construct_kwargs['buckets'] = buckets
        _prometheus_metrics[prom_name] = _prom_histogram(
            prom_name, description, **construct_kwargs
        )
      self._metric = _prometheus_metrics[prom_name]

  def _GetPrometheusBuckets(
      self, bucketer: Bucketer, units: Units | None = None
  ) -> list[float] | None:
    """Generates Prometheus buckets from the bucketer definition."""
    if bucketer.type != 'PowersOf' or not bucketer.args:
      return None

    base = bucketer.args[0]
    if base <= 1.0:
      logging.warning(
          'Prometheus Bucketer.PowersOf requires base > 1.0, got %f', base
      )
      return None

    # Determine the maximum value for the buckets.
    # Default to a very large number if units are not time-based.
    max_val = float('inf')
    start_val = 1.0
    if units == Units.SECONDS:
      max_val = 3600.0
      # Start at 1 millisecond for seconds-based metrics.
      start_val = 0.001
    elif units == Units.MILLISECONDS or units is None:
      # Default to milliseconds (3.6M) if units are MILLISECONDS or None.
      max_val = 3600000.0
      start_val = 1.0
    elif units == Units.MICROSECONDS:
      max_val = 3600000000.0
      start_val = 1.0
    elif units == Units.NANOSECONDS:
      max_val = 3600000000000.0
      start_val = 1.0

    buckets = []
    val = start_val
    # Add a safeguard to prevent excessively many buckets, even with inf
    # max_val.
    # A practical limit for Prometheus histograms is around 1e18.
    practical_limit = 1e18
    while val < min(max_val, practical_limit):
      buckets.append(val)
      val *= base
    return buckets

  def Record(self, value, *args, **kwargs):
    """Records a value in the Prometheus histogram."""
    if not self._metric:
      return
    if not args and not kwargs:
      self._metric.observe(value)
      return
    try:
      self._metric.labels(*args, **kwargs).observe(value)
    except ValueError as e:
      logging.warning(
          'Failed to record Prometheus histogram due to label mismatch: %s.',
          e,
      )

  def Get(self, *args, **kwargs):
    """Returns 0.0 as Prometheus histograms do not support Get."""
    del args, kwargs
    return 0.0

  def ClearAll(self):
    """No-op for Prometheus histograms."""
    pass


def Initialize(port=None):
  """Initializes PyGrain metric reporting."""
  global _initialized
  if _initialized:
    return

  if not prometheus_client:
    return

  if port is None:
    env_port = os.environ.get('PYGRAIN_PROMETHEUS_PORT')
    port = _DEFAULT_PROMETHEUS_PORT
    if env_port:
      try:
        port = int(env_port)
      except ValueError:
        logging.warning(
            'Invalid PYGRAIN_PROMETHEUS_PORT "%s". Falling back to default %d.',
            env_port,
            _DEFAULT_PROMETHEUS_PORT,
        )
  if not prometheus_client:
    logging.warning(
        'prometheus-client not found. Grain metrics will not be reported to'
        ' Prometheus.'
    )
    return

  with _lock:
    if _initialized:
      return
    if port <= 0:
      _initialized = True
      return

    try:
      # If multiprocess directory is configured, use MultiProcessCollector
      # to aggregate metrics from all worker processes.
      multiprocess_started = False
      if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        try:
          registry = prometheus_client.CollectorRegistry()
          multiprocess.MultiProcessCollector(registry)
          prometheus_client.start_http_server(port, registry=registry)
          logging.info(
              'Prometheus multiprocess metrics server started on port %s.',
              port,
          )
          multiprocess_started = True
        except (ImportError, AttributeError):
          pass

      if not multiprocess_started:
        # Standard single-process server
        prometheus_client.start_http_server(port)
        logging.info('Prometheus metrics server started on port %s.', port)
    except ValueError as e:
      logging.warning('Failed to start Prometheus server: %s', e)
      return
    except OSError as e:
      if e.errno != errno.EADDRINUSE:
        logging.warning('Failed to start Prometheus server: %s', e)
        return
      logging.info('Prometheus server already active.')

    _initialized = True


def SetupTelemetry():
  """Autostarts Prometheus metrics server if enabled via environment variable."""
  enable_telemetry = os.environ.get(
      'ENABLE_PYGRAIN_PROMETHEUS_TELEMETRY', 'false'
  )
  if enable_telemetry.lower() == 'true':
    global _prometheus_multiproc_dir

    if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
      # Create a directory for prometheus multiprocessing.
      _prometheus_multiproc_dir = tempfile.mkdtemp(
          prefix='prometheus_multiproc_'
      )
      os.environ['PROMETHEUS_MULTIPROC_DIR'] = _prometheus_multiproc_dir
      _creator_pid = os.getpid()

      def _Cleanup():
        if os.getpid() == _creator_pid:
          shutil.rmtree(_prometheus_multiproc_dir, ignore_errors=True)

      atexit.register(_Cleanup)

    if multiprocessing.current_process().name == 'MainProcess':
      Initialize(port=9431)
    else:
      Initialize(port=0)


_bytes_read = Counter(
    '/grain/python/data_sources/bytes_read',
    Metadata(
        description=(
            'Number of bytes produced by a data source via random access.'
        )
    ),
    fields=[('source', str)],
)

_source_read_time_ns = EventMetric(
    '/grain/python/dataset/source_read_time_ns',
    metadata=Metadata(
        description='Histogram of source read time in nanoseconds.',
        units=Units.NANOSECONDS,
    ),
    bucketer=Bucketer.PowersOf(4.0),
    fields=[('source', str)],
)


def RecordBytesReadAndLatency(
    source: str, num_bytes: int, latency_ns: int, num_reads: int
):
  """Records the number of bytes read and read latency for a Grain source."""
  _bytes_read.IncrementBy(num_bytes, source)
  for _ in range(num_reads):
    _source_read_time_ns.Record(latency_ns / num_reads, source)


record_bytes_read_and_latency = RecordBytesReadAndLatency
