# Copyright 2024 Google LLC
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
import re

from absl.testing import absltest
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset import visualize
import numpy as np


_MAP_DATASET_REPR = r"""RangeMapDataset(start=0, stop=10, step=1)
  ││
  ││  
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  ShuffleMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapWithIndexMapDataset(transform=_add_dummy_metadata @ .../python/dataset/visualize_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  MapMapDataset(transform=_identity @ .../python/dataset/visualize_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}
"""

_ITER_DATASET_REPR = r"""RangeMapDataset(start=0, stop=10, step=1)
  ││
  ││  
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  ShuffleMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  PrefetchIterDataset(read_options=ReadOptions(num_threads=16, prefetch_buffer_size=500), allow_nones=False)
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapIterDataset(transform=<lambda> @ .../python/dataset/visualize_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  BatchIterDataset(batch_size=2, drop_remainder=False)
  ││
  ╲╱
{'data': 'int64[2]',
 'dataset_index': 'int64[2]',
 'epoch': 'int64[2]',
 'index': 'int64[2]'}
"""

_MIX_DATASET_REPR = r"""WARNING: Detected multi-parent datasets: MixedMapDataset[2 parents]. Only displaying the first parent.

RangeMapDataset(start=0, stop=10, step=1)
  ││
  ││  
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  ShuffleMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MixedMapDataset[2 parents]
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapMapDataset(transform=_AddOne)
  ││
  ╲╱
"<class 'int'>[]"
"""

_PREFETCH_DATASET_REPR = r"""RangeMapDataset(start=0, stop=10, step=1)
  ││
  ││  
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  ShuffleMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapWithIndexMapDataset(transform=_add_dummy_metadata @ .../python/dataset/visualize_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  PrefetchIterDataset(read_options=ReadOptions(num_threads=16, prefetch_buffer_size=500), allow_nones=False)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  MultiprocessPrefetchIterDataset(multiprocessing_options=MultiprocessingOptions(num_workers=1, per_worker_buffer_size=1, enable_profiling=False))
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}
"""

_SOURCE_DATASET_REPR = r"""SourceMapDataset(source=_ExpensiveSource)
  ││
  ││  
  ││
  ╲╱
"<class 'bytes'>[]"

  ││
  ││  MapMapDataset(transform=_identity @ .../python/dataset/visualize_test.py:XXX)
  ││
  ╲╱
"<class 'bytes'>[]"
"""


def _deep_dataset_repr(ds):
  return [repr(ds)] + [_deep_dataset_repr(p) for p in ds.parents]


def _add_dummy_metadata(i, x):
  return {"data": x, "index": i, "epoch": 4, "dataset_index": 1}


def _identity(x):
  return x


class _AddOne(transforms.MapTransform):

  def map(self, x):
    return x + 1


class _ExpensiveSource:
  data_was_touched = False

  def __len__(self):
    return 10

  def __getitem__(self, idx):
    _ExpensiveSource.data_was_touched = True
    return b"this should not be returned"


class VisualizeTest(absltest.TestCase):

  def _assert_visualization(self, ds, expected):
    original_repr = _deep_dataset_repr(ds)
    original_result = list(ds)
    result = visualize._build_visualization_str(ds, None)
    print(result)
    # Remove line number from the result to make test less brittle.
    result = re.sub(r".py:\d+", ".py:XXX", result)
    self.assertEqual(result, expected)
    repr_after_visualize = _deep_dataset_repr(ds)
    result_after_visualize = list(ds)
    self.assertEqual(original_repr, repr_after_visualize)
    np.testing.assert_equal(original_result, result_after_visualize)

  def test_visualize_map(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
    )
    self._assert_visualization(ds, _MAP_DATASET_REPR)

  def test_visualize_iter(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .to_iter_dataset()
        .map(lambda x: _add_dummy_metadata(2, x))
        .batch(2)
    )
    self._assert_visualization(ds, _ITER_DATASET_REPR)

  def test_visualize_with_mix(self):
    ds1 = dataset.MapDataset.range(10).shuffle(42)
    ds2 = dataset.MapDataset.range(10).shuffle(43)
    ds = dataset.MapDataset.mix([ds1, ds2]).map(_AddOne())
    self._assert_visualization(ds, _MIX_DATASET_REPR)

  def test_visualize_with_mp_prefetch(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .map_with_index(_add_dummy_metadata)
        .to_iter_dataset()
        .mp_prefetch(options.MultiprocessingOptions(num_workers=1))
    )
    self._assert_visualization(ds, _PREFETCH_DATASET_REPR)

  def test_visualize_with_source_mock(self):
    ds = dataset.MapDataset.source(_ExpensiveSource()).map(_identity)
    # Make sure the actual data was not read.
    _ = visualize._build_visualization_str(ds, b"mock source output")
    self.assertFalse(_ExpensiveSource.data_was_touched)
    # Note that the assertion below will iterate over the dataset to assert
    # its integrity after visualization, so the `data_was_touched` will be
    # flipped.
    self._assert_visualization(ds, _SOURCE_DATASET_REPR)


if __name__ == "__main__":
  absltest.main()
