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
"""Experimental Grain APIs.

Backwards compatibility re-import. Prefer adding new APIs to top-level
`experimental.py` file.
"""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-multiple-import
# pylint: disable=unused-import

from grain._src.python.dataset.base import (
    DatasetOptions,
    ExecutionTrackingMode,
)
from grain._src.python.dataset.dataset import (
    apply_transformations,
    WithOptionsIterDataset,
)
from grain._src.python.dataset.sources.parquet_dataset import ParquetIterDataset

from grain._src.python.dataset.transformations.flatmap import (
    FlatMapMapDataset,
    FlatMapIterDataset,
)
from grain._src.python.dataset.transformations.interleave import (
    InterleaveIterDataset,
)
from grain._src.python.dataset.transformations.limit import LimitIterDataset
from grain._src.python.dataset.transformations.map import RngPool
from grain._src.python.dataset.transformations.mix import ConcatenateMapDataset
from grain._src.python.dataset.transformations.packing import FirstFitPackIterDataset
from grain._src.python.dataset.transformations.packing_concat_then_split import (
    BOSHandling,
    ConcatThenSplitIterDataset,
)
from grain._src.python.dataset.transformations.prefetch import (
    MultiprocessPrefetchIterDataset,
    ThreadPrefetchIterDataset,
)
from grain._src.python.dataset.transformations.shuffle import (
    WindowShuffleMapDataset,
    WindowShuffleIterDataset,
)
from grain._src.python.dataset.transformations.zip import (
    ZipMapDataset,
    ZipIterDataset,
)
from grain._src.core.transforms import (
    FlatMapTransform,
    MapWithIndex as MapWithIndexTransform,
)
from grain._src.python.experimental.example_packing.packing import PackAndBatchOperation

# This should evetually live under grain.testing.
from grain._src.python.testing.experimental import assert_equal_output_after_checkpoint

from grain._src.python.experimental.index_shuffle.python.index_shuffle_module import index_shuffle
