# Copyright 2022 Google LLC
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
"""APIs for Grain Python backend."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import

from . import python_experimental as experimental

from ._src.core.config import config
from ._src.core.sharding import NoSharding, ShardByJaxProcess, ShardOptions
from ._src.core.transforms import (
    BatchTransform as Batch,
    FilterTransform,
    MapTransform,
    RandomMapTransform,
    Transformations,
    Transformation,
)

from ._src.python.checkpoint_handlers import PyGrainCheckpointHandler
from ._src.python.data_loader import (
    DataLoader,
    PyGrainDatasetIterator,
)
from ._src.python.data_sources import (
    ArrayRecordDataSource,
    InMemoryDataSource,
    RandomAccessDataSource,
    RangeDataSource,
)
from ._src.python.grain_pool import (GrainPool, GrainPoolElement)
from ._src.python.load import load
from ._src.python.operations import (
    BatchOperation,
    FilterOperation,
    MapOperation,
    Operation,
    RandomMapOperation,
)
from ._src.python.options import ReadOptions
from ._src.python.record import (Record, RecordMetadata)
from ._src.python.samplers import (
    IndexSampler,
    Sampler,
    SequentialSampler,
)
from ._src.python.shared_memory_array import SharedMemoryArray
