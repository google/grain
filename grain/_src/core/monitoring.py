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
"""Grain metrics convenience functions and implementation selection."""

from __future__ import annotations

from grain._src.core import monitoring_base
from grain._src.core import prometheus_monitoring as _impl
fast_monitoring = None


def record_bytes_read_and_latency(
    source: str, *, num_bytes: int, latency_ns: int, num_reads: int
) -> None:
  """Records the number of bytes read and read latency for a Grain source."""
  if fast_monitoring is not None:
    fast_monitoring.record_bytes_read_and_latency(
        source, num_bytes, latency_ns, num_reads
    )
  else:
    _impl.record_bytes_read_and_latency(
        source, num_bytes=num_bytes, latency_ns=latency_ns, num_reads=num_reads
    )


Bucketer = monitoring_base.Bucketer
Metadata = monitoring_base.Metadata
Units = monitoring_base.Units

Counter = _impl.Counter
EventMetric = _impl.EventMetric
Metric = _impl.Metric
get_monitoring_root = _impl.get_monitoring_root
setup_telemetry = _impl.SetupTelemetry


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
