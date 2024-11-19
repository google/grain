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
# pylint: disable=g-bad-import-order
# pylint: disable=g-multiple-import
# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=g-import-not-at-top

from etils import epy

# `lazy_dataset` module is deprecated. It displays the deprecation message upon
# import. We make the import lazy to only display the message when the module
# is actually used.
with epy.lazy_imports():
  from grain import python_lazy_dataset as lazy_dataset

del epy

from ._src.python.dataset.dataset import (
    apply_transformations,
    DatasetOptions,
    WithOptionsIterDataset,
)
from ._src.python.dataset.transformations.flatmap import (
    FlatMapMapDataset,
    FlatMapIterDataset,
)
from ._src.python.dataset.transformations.interleave import (
    InterleaveIterDataset,
)
from ._src.python.dataset.transformations.map import RngPool
from ._src.python.dataset.transformations.mix import ConcatenateMapDataset
from ._src.python.dataset.transformations.packing import FirstFitPackIterDataset
from ._src.python.dataset.transformations.prefetch import (
    MultiprocessPrefetchIterDataset,
    ThreadPrefetchIterDataset,
)
from ._src.python.dataset.transformations.shuffle import WindowShuffleMapDataset
from ._src.python.dataset.transformations.zip import ZipMapDataset
from ._src.core.transforms import (
    FlatMapTransform,
    MapWithIndexTransform,
)
from ._src.python.experimental.example_packing.packing import PackAndBatchOperation
from ._src.python.experimental.index_shuffle.python.index_shuffle_module import index_shuffle
