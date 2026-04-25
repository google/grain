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

import enum
import logging
import threading
from typing import Any

import os
import tempfile

if os.environ.get('ENABLE_PYGRAIN_TELEMETRY', 'false').lower() == 'true':
  if 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
    os.environ['PROMETHEUS_MULTIPROC_DIR'] = tempfile.mkdtemp(
        prefix='grain_metrics_'
    )

import importlib

try:
  prometheus_client = importlib.import_module('prometheus_client')

  _prom_counter = prometheus_client.Counter  # pytype: disable=attribute-error
  _prom_gauge = prometheus_client.Gauge  # pytype: disable=attribute-error
  _prom_histogram = prometheus_client.Histogram  # pytype: disable=attribute-error
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
    '/grain/python/next_duration_ns',
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
    """Increments the counter by 1.

    Args:
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.
    """
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args or kwargs:
          self._metric.labels(*args, **kwargs).inc()  # pytype: disable=attribute-error
        else:
          self._metric.inc()  # pytype: disable=attribute-error

  def IncrementBy(self, value, *args, **kwargs):
    """Increments the counter by a specific value.

    Args:
      value: the value to increment by.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.
    """
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args or kwargs:
          self._metric.labels(*args, **kwargs).inc(value)  # pytype: disable=attribute-error
        else:
          self._metric.inc(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the counter.

    Args:
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      In OSS this always returns 0.0.
    """
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
    """Sets the metric to a specific value.

    Args:
      value: the value to set.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.
    """
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args or kwargs:
          self._metric.labels(*args, **kwargs).set(value)  # pytype: disable=attribute-error
        else:
          self._metric.set(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the metric.

    Args:
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      In OSS this always returns 0.0.
    """
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
        units = getattr(_metadata, 'units', None)
        buckets = bucketer.get_prometheus_buckets(units=units)

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
    """Records an observation for the event metric.

    Args:
      value: the value to record.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.
    """
    if not self._use_prometheus:
      pass
    else:
      if self._metric:
        if args or kwargs:
          self._metric.labels(*args, **kwargs).observe(value)  # pytype: disable=attribute-error
        else:
          self._metric.observe(value)  # pytype: disable=attribute-error

  def Get(self, *args, **kwargs):
    """Returns the current value of the event metric.

    Args:
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      In OSS this always returns 0.0.
    """
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

  def get_prometheus_buckets(
      self, units: Units | None = None
  ) -> list[float] | None:
    """Generates Prometheus buckets from the bucketer definition."""
    if self._type == 'PowersOf' and self._args:
      base = self._args[0]
      if base <= 1.0:
        logging.warning(
            'Prometheus Bucketer.PowersOf requires base > 1.0, got %f', base
        )
        return None

      # Scale the 1-hour limit based on units.
      # Default to milliseconds (3.6M) if units are unknown.
      max_val = 3600000.0
      if units == Units.SECONDS:
        max_val = 3600.0
      elif units == Units.MICROSECONDS:
        max_val = 3600000000.0
      elif units == Units.NANOSECONDS:
        max_val = 3600000000000.0

      buckets = []
      val = 1.0
      while val < max_val:
        buckets.append(val)
        val *= base
      return buckets
    return None


def record_autotune_node_throughput(node_name: str, throughput: float):
  _grain_autotune_node_throughput.Set(throughput, node_name)


def record_autotune_parameter(node_name: str, name: str, value: float):
  _grain_autotune_parameters.Set(value, node_name, name)


def record_autotune_usl_coeff(node_name: str, name: str, value: float):
  _grain_autotune_usl_coeffs.Set(value, node_name, name)


def record_autotune_optimization_latency(latency_ms: float):
  _grain_autotune_optimization_latency.Record(latency_ms)


def record_framework_type(framework_type: str):
  _grain_framework_type_metric.Increment(framework_type)


def set_debug_server_ports(ports: list[int]):
  if not ports:
    return
  _grain_debug_server_ports.Set(','.join(map(str, ports)))


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


def initialize(port=9431):
  """Initializes PyGrain metric reporting."""
  global _initialized
  if _initialized:
    return
  if not _USE_PROMETHEUS:
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

    if port > 0:
      try:
        # If multiprocess directory is configured, use MultiProcessCollector
        # to aggregate metrics from all worker processes.
        multiprocess_started = False
        if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
          try:
            multiprocess = importlib.import_module(
                'prometheus_client.multiprocess'
            )
            registry = prometheus_client.CollectorRegistry()  # pytype: disable=attribute-error
            multiprocess.MultiProcessCollector(registry)  # pytype: disable=attribute-error
            prometheus_client.start_http_server(port, registry=registry)  # pytype: disable=attribute-error
            logging.info(
                'Prometheus multiprocess metrics server started on port %s.',
                port,
            )
            multiprocess_started = True
          except (ImportError, AttributeError):
            pass

        if not multiprocess_started:
          # Standard single-process server
          prometheus_client.start_http_server(port)  # pytype: disable=attribute-error
          logging.info('Prometheus metrics server started on port %s.', port)
      except (OSError, ValueError) as e:
        # Handle 'already in use' for Linux/macOS and Windows (10048).
        if 'already in use' not in str(e) and '10048' not in str(e):
          logging.warning('Failed to start Prometheus server: %s', e)
          return
          # If the server is already running (e.g. started by Orbax), just
          # register listeners.
        logging.info('Prometheus server already active.')

    _initialized = True


def setup_telemetry():
  """Autostarts Prometheus metrics server if enabled via environment variable."""
  if os.environ.get('ENABLE_PYGRAIN_TELEMETRY', 'false').lower() == 'true':
    import multiprocessing

    if multiprocessing.current_process().name == 'MainProcess':
      initialize(port=9431)
    else:
      initialize(port=0)
