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
"""Public API for Grain.

Backwards compatibility re-import. Prefer adding new APIs to top-level modules.
"""

# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
# pylint: disable=g-multiple-import
# pylint: disable=unused-import


from grain._src.core.config import config
from grain._src.core.constants import (
    DATASET_INDEX,
    EPOCH,
    INDEX,
    META_FEATURES,
    RECORD,
    RECORD_KEY,
    SEED,
)
from grain._src.core.sharding import NoSharding, ShardByJaxProcess, ShardOptions
from grain._src.core.transforms import (
    Batch,
    Filter as FilterTransform,
    MapTransform,
    MapWithIndex as MapWithIndexTransform,
    RandomMapTransform,
    Transformation,
    Transformations,
)

from grain._src.python.checkpoint_handlers import (
    CheckpointHandler as PyGrainCheckpointHandler,
)
from grain._src.python.data_loader import (
    DataLoader,
    DataLoaderIterator as PyGrainDatasetIterator,
)
from grain._src.python.data_sources import (
    ArrayRecordDataSource,
    SharedMemoryDataSource as InMemoryDataSource,
    RandomAccessDataSource,
    RangeDataSource,
)
from grain._src.python.dataset.base import DatasetSelectionMap
from grain._src.python.dataset.dataset import (
    MapDataset,
    IterDataset,
    DatasetIterator,
)

from grain._src.python.load import load
from grain._src.python.operations import (
    BatchOperation,
    FilterOperation,
    MapOperation,
    Operation,
    RandomMapOperation,
)
from grain._src.python.options import ReadOptions, MultiprocessingOptions
from grain._src.python.record import (Record, RecordMetadata)
from grain._src.python.samplers import (
    IndexSampler,
    Sampler,
    SequentialSampler,
)
from grain._src.python.shared_memory_array import SharedMemoryArray
from grain.python import experimental

# These are imported only if Orbax is present.
try:
  from grain._src.python.checkpoint_handlers import (
      CheckpointSave as PyGrainCheckpointSave,
      CheckpointRestore as PyGrainCheckpointRestore,
  )
except ImportError:
  pass
