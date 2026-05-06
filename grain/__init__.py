# Copyright 2025 Google LLC
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
"""Public API for Grain."""

# pylint: disable=g-importing-member
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
# pylint: disable=g-import-not-at-top

import os

# We import all public modules here to enable the use of `grain.foo.Bar`
# instead of forcing users to write `from grain import foo as grain_foo`.
from grain import (
    experimental,
    checkpoint,
    constants,
    multiprocessing,
    samplers,
    sharding,
    sources,
    transforms,
)

from grain._src.core import monitoring as _grain_monitoring
from grain._src.core.config import config
from grain._src.core.version import __version__, __version_info__
from grain._src.python.data_loader import (
    DataLoader,
    DataLoaderIterator,
)
from grain._src.python.dataset.dataset import (
    DatasetIterator,
    IterDataset,
    MapDataset,
)
from grain._src.python.load import load
from grain._src.python.options import ReadOptions, MultiprocessingOptions
from grain._src.python.record import Record, RecordMetadata

# Autostart Prometheus metrics server if not disabled.
# Default port is 8000. Use environment variable to override or set to 0 to
# disable.
try:
  _prometheus_port = int(os.environ.get('GRAIN_PROMETHEUS_PORT', 8000))
except ValueError:
  _prometheus_port = 8000
if _prometheus_port > 0:
  _grain_monitoring.initialize(port=_prometheus_port)
