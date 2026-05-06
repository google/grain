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
"""Grain metrics."""

from __future__ import annotations

import enum
import logging
import os
import threading
from typing import Any

try:
  import prometheus_client  # pytype: disable=import-error

  _prom_counter = prometheus_client.Counter
  _prom_gauge = prometheus_client.Gauge
  _prom_histogram = prometheus_client.Histogram
except (ImportError, AttributeError):
  prometheus_client = None
  _prom_counter = None
  _prom_gauge = None
  _prom_histogram = None

# pylint: disable=invalid-name

_USE_PROMETHEUS = True

_initialized = False
_prometheus_metrics = {}  # Maps metric names to their Prometheus objects.
_lock = threading.Lock()

_PROMETHEUS_ALLOWED_METRICS = {
    '/grain/python/dataset/next_duration_ns',
}


@enum.unique
class Units(enum.Enum):
  """Grain metric units."""

  SECONDS = enum.auto()
  MILLISECONDS = enum.auto()
  MICROSECONDS = enum.auto()
  NANOSECONDS = enum.auto()
  BITS = enum.auto()
  BYTES = enum.auto()


def _is_allowed(metric_name: str) -> bool:
  """Returns True if the metric is allowed for Prometheus export."""
  return metric_name in _PROMETHEUS_ALLOWED_METRICS


# pylint: disable=function-redefined
def get_monitoring_root() -> None:
  """Fallback monitoring root."""
  return None


# pylint: enable=function-redefined


class Metadata:
  """Grain metric metadata."""

  def __init__(self, description='', **kwargs):
    self.description = description
    for key, value in kwargs.items():
      setattr(self, key, value)
    self._kwargs = kwargs


_METADATA_TYPES = (Metadata,)


def _extract_metadata_and_fields(
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
  """Unified wrapper for metric emission."""

  def __init__(self, name, metadata=None, *args, **kwargs):
    _metadata, _fields, _args = _extract_metadata_and_fields(
        metadata, args, kwargs
    )

    self._use_prometheus = _USE_PROMETHEUS
    if not self._use_prometheus:
      pass
    else:
      if not _is_allowed(name) or not _prom_counter:
        self._metric = None
        return

      prom_name = name.strip('/').replace('/', '_')
      labelnames = tuple(f[0] for f in _fields)
      description = 'Grain Counter'
      if _metadata and hasattr(_metadata, 'description'):
        description = _metadata.description

      with _lock:
        if prom_name not in _prometheus_metrics:
          if _prom_counter:
            _prometheus_metrics[prom_name] = _prom_counter(
                prom_name, description, labelnames=labelnames
            )
          else:
            _prometheus_metrics[prom_name] = None
        self._metric = _prometheus_metrics[prom_name]

  def Increment(self, *args, **kwargs):
    """Increments the counter."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args:
          self._metric.labels(*args).inc()  # pytype: disable=attribute-error
        else:
          self._metric.inc()  # pytype: disable=attribute-error

  def IncrementBy(self, value, *args, **kwargs):
    """Increments the counter by a specific value."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args:
          self._metric.labels(*args).inc(value)  # pytype: disable=attribute-error
        else:
          self._metric.inc(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the counter."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    return 0.0

  def ClearAll(self):
    """Clears all values for this metric."""
    if not self._use_prometheus:
      pass


class Metric:
  """Unified wrapper for metric emission."""

  def __init__(self, name, value_type=float, metadata=None, *args, **kwargs):
    # For Metric, name and value_type are positional. metadata is next.
    _metadata, _fields, _args = _extract_metadata_and_fields(
        metadata, args, kwargs
    )

    self._use_prometheus = _USE_PROMETHEUS
    if not self._use_prometheus:
      pass
    else:
      if not _is_allowed(name) or not _prom_gauge:
        self._metric = None
        return

      prom_name = name.strip('/').replace('/', '_')
      labelnames = tuple(f[0] for f in _fields)
      description = 'Grain Gauge'
      if _metadata and hasattr(_metadata, 'description'):
        description = _metadata.description

      with _lock:
        if prom_name not in _prometheus_metrics:
          if _prom_gauge:
            _prometheus_metrics[prom_name] = _prom_gauge(
                prom_name, description, labelnames=labelnames
            )
          else:
            _prometheus_metrics[prom_name] = None
        self._metric = _prometheus_metrics[prom_name]

  def Set(self, value, *args, **kwargs):
    """Sets the metric value."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args:
          self._metric.labels(*args).set(value)  # pytype: disable=attribute-error
        else:
          self._metric.set(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the metric."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    return 0.0

  def ClearAll(self):
    """Clears all values for this metric."""
    if not self._use_prometheus:
      pass


class EventMetric:
  """Unified wrapper for metric emission."""

  def __init__(self, name, metadata=None, *args, **kwargs):
    _metadata, _fields, _args = _extract_metadata_and_fields(
        metadata, args, kwargs
    )

    self._use_prometheus = _USE_PROMETHEUS
    if not self._use_prometheus:
      pass
    else:
      if not _is_allowed(name) or not _prom_histogram:
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
        buckets = bucketer.get_prometheus_buckets()

      with _lock:
        if prom_name not in _prometheus_metrics:
          if _prom_histogram:
            construct_kwargs = {'labelnames': labelnames}
            if buckets:
              construct_kwargs['buckets'] = buckets
            _prometheus_metrics[prom_name] = _prom_histogram(
                prom_name, description, **construct_kwargs
            )
          else:
            _prometheus_metrics[prom_name] = None
        self._metric = _prometheus_metrics[prom_name]

  def Record(self, value, *args, **kwargs):
    """Records an observation."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args:
          self._metric.labels(*args).observe(
              value
          )  # pytype: disable=attribute-error
        else:
          self._metric.observe(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the event metric."""
    del kwargs  # Unused.
    if not self._use_prometheus:
      pass
    return 0.0

  def ClearAll(self):
    """Clears all values for this metric."""
    if not self._use_prometheus:
      pass


class Bucketer:
  """Grain metric bucketer."""

  def __init__(self, *args, bucketer_type=None, **kwargs):
    self._args = args
    self._kwargs = kwargs
    self._type = bucketer_type

  @staticmethod
  def PowersOf(base: float):
    return Bucketer(base, bucketer_type='PowersOf')

  def get_prometheus_buckets(self) -> list[float] | None:
    """Generates Prometheus buckets from the bucketer definition."""
    if self._type == 'PowersOf' and self._args:
      base = self._args[0]
      if base <= 1.0:
        logging.warning(
            'Prometheus Bucketer.PowersOf requires base > 1.0, got %f', base
        )
        return None
      # Generate buckets up to 1 hour in milliseconds (3.6M).
      buckets = []
      val = 1.0
      while val < 3600000.0:
        buckets.append(val)
        val *= base
      return buckets
    return None


_grain_framework_type_metric = Counter(
    '/grain/framework_type',
    Metadata(description='The framework type used to build the Grain dataset.'),
    fields=[('name', str)],
)

_grain_debug_server_ports = Metric(
    '/grain/debug_server_ports',
    str,
    metadata=Metadata(
        description=(
            'A comma-separated list of debug server ports. The first port is'
            ' for the main process, followed by worker process ports.'
        )
    ),
)


_grain_autotune_node_throughput = Metric(
    '/grain/autotune/node_throughput',
    float,
    metadata=Metadata(
        description='The observed throughput of the autotune node (elements/s).'
    ),
    fields=[('node_name', str)],
)

_grain_autotune_parameters = Metric(
    '/grain/autotune/parameters',
    float,
    metadata=Metadata(
        description='The current values for the autotuned parameters.'
    ),
    fields=[('node_name', str), ('name', str)],
)

_grain_autotune_usl_coeffs = Metric(
    '/grain/autotune/usl_coeffs',
    float,
    metadata=Metadata(
        description='The estimated USL coefficients for the autotune node.'
    ),
    fields=[('node_name', str), ('name', str)],
)

_grain_autotune_optimization_latency = EventMetric(
    '/grain/autotune/optimization_latency',
    metadata=Metadata(
        description='The time taken by the autotune optimizer (ms).'
    ),
)


def record_autotune_node_throughput(node_name: str, throughput: float):
  """Records the throughput of an autotune node."""
  _grain_autotune_node_throughput.Set(throughput, node_name)


def record_autotune_parameter(node_name: str, name: str, value: float):
  """Records an autotune parameter value."""
  _grain_autotune_parameters.Set(value, node_name, name)


def record_autotune_usl_coeff(node_name: str, name: str, value: float):
  """Records an autotune USL coefficient."""
  _grain_autotune_usl_coeffs.Set(value, node_name, name)


def record_autotune_optimization_latency(latency_ms: float):
  """Records the latency of the autotune optimization."""
  _grain_autotune_optimization_latency.Record(latency_ms)


def record_framework_type(framework_type: str):
  """Records the framework type."""
  _grain_framework_type_metric.Increment(framework_type)


def set_debug_server_ports(ports: list[int]):
  """Sets the debug server ports streamz metric."""
  if not ports:
    return
  _grain_debug_server_ports.Set(','.join(map(str, ports)))


def initialize(port=8000):
  """Initializes PyGrain metric reporting."""
  global _initialized
  if _initialized:
    return
  if not _USE_PROMETHEUS:
    return
  if os.environ.get('DISABLE_PYGRAIN_TELEMETRY', 'false').lower() == 'true':
    logging.info('PyGrain telemetry is deactivated via environment variable.')
    return

  if not prometheus_client:
    logging.warning(
        'prometheus-client not found. Grain metrics will not be reported to'
        ' Prometheus.'
    )
    return

  with _lock:
    if _initialized:
      return
    try:
      if port > 0:
        prometheus_client.start_http_server(port)
        logging.info('Prometheus metrics server started on port %s.', port)
      _initialized = True
    except (OSError, ValueError) as e:
      # Handle 'already in use' for Linux/macOS and Windows (10048).
      if 'already in use' in str(e) or '10048' in str(e):
        _initialized = True
        logging.info('Prometheus server already active.')
      else:
        logging.warning('Failed to initialize Prometheus metrics: %s', e)
