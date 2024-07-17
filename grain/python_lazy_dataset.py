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

from ._src.python.lazy_dataset.data_loader import DataLoader
from ._src.python.lazy_dataset.data_sources import (
    SourceMapDataset as SourceLazyMapDataset,
    log_lineage_for_sources,
)
from ._src.python.lazy_dataset.lazy_dataset import (
    MapDataset as LazyMapDataset,
    IterDataset as LazyIterDataset,
    DatasetIterator as LazyDatasetIterator,
    RangeMapDataset as RangeLazyMapDataset,
)
from ._src.python.lazy_dataset.transformations.batch import (
    BatchMapDataset as BatchLazyMapDataset,
    BatchIterDataset as BatchLazyIterDataset,
)
from ._src.python.lazy_dataset.transformations.filter import (
    FilterMapDataset as FilterLazyMapDataset,
    FilterIterDataset as FilterLazyIterDataset,
)
from ._src.python.lazy_dataset.transformations.flatmap import FlatMapMapDataset
from ._src.python.lazy_dataset.transformations.map import (
    MapMapDataset as MapLazyMapDataset,
    MapIterDataset as MapLazyIterDataset,
    RngPool,
)
from ._src.python.lazy_dataset.transformations.mix import (
    ConcatenateMapDataset as ConcatenateLazyMapDataset,
    DatasetSelectionMap,
    MixedMapDataset as MixedLazyMapDataset,
    MixedIterDataset as MixedLazyIterDataset,
)
from ._src.python.lazy_dataset.transformations.packing import (
    FirstFitPackIterDataset as FirstFitPackLazyIterDataset,
    SingleBinPackIterDataset as SingleBinPackLazyIterDataset,
)
from ._src.python.lazy_dataset.transformations.prefetch import ThreadPrefetchIterDataset
from ._src.python.lazy_dataset.transformations.repeat import RepeatMapDataset
from ._src.python.lazy_dataset.transformations.shuffle import (
    ShuffleMapDataset as ShuffleLazyMapDataset,
    WindowShuffleMapDataset as WindowShuffleLazyMapDataset,
)
from ._src.python.lazy_dataset.transformations.slice import (
    SliceMapDataset as SliceLazyMapDataset,
)
from ._src.python.lazy_dataset.transformations.zip import (
    ZipMapDataset as ZipLazyMapDataset,
)
