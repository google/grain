# Copyright 2023 Google LLC
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
"""Experimental PyGrain APIs."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import
from ._src.python.dataset.base import DatasetSelectionMap
from ._src.python.dataset.data_loader import DataLoader
from ._src.python.dataset.dataset import (
    MapDataset as LazyMapDataset,
    IterDataset as LazyIterDataset,
    DatasetIterator as LazyDatasetIterator,
)
from ._src.python.dataset.transformations.batch import (
    BatchMapDataset as BatchLazyMapDataset,
    BatchIterDataset as BatchLazyIterDataset,
)
from ._src.python.dataset.transformations.filter import (
    FilterMapDataset as FilterLazyMapDataset,
    FilterIterDataset as FilterLazyIterDataset,
)
from ._src.python.dataset.transformations.flatmap import (
    FlatMapMapDataset as FlatMapLazyMapDataset,
    FlatMapIterDataset,
)
from ._src.python.dataset.transformations.map import (
    MapMapDataset as MapLazyMapDataset,
    MapIterDataset as MapLazyIterDataset,
    RngPool,
)
from ._src.python.dataset.transformations.mix import (
    ConcatenateMapDataset as ConcatenateLazyMapDataset,
    MixedMapDataset as MixedLazyMapDataset,
    MixedIterDataset as MixedLazyIterDataset,
)
from ._src.python.dataset.transformations.packing import (
    FirstFitPackIterDataset as FirstFitPackLazyIterDataset,
    SingleBinPackIterDataset as SingleBinPackLazyIterDataset,
)
from ._src.python.dataset.transformations.prefetch import (
    ThreadPrefetchIterDataset as ThreadPrefetchLazyIterDataset,
)
from ._src.python.dataset.transformations.repeat import (
    RepeatMapDataset as RepeatLazyMapDataset,
)
from ._src.python.dataset.transformations.shuffle import (
    ShuffleMapDataset as ShuffleLazyMapDataset,
    WindowShuffleMapDataset as WindowShuffleLazyMapDataset,
)
from ._src.python.dataset.transformations.slice import (
    SliceMapDataset as SliceLazyMapDataset,
)
from ._src.python.dataset.transformations.source import (
    RangeMapDataset as RangeLazyMapDataset,
    SourceMapDataset as SourceLazyMapDataset,
    log_lineage_for_sources,
)
from ._src.python.dataset.transformations.zip import (
    ZipMapDataset as ZipLazyMapDataset,
)
