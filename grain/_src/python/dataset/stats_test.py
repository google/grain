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

import collections
import functools
import threading
import time
from unittest import mock

from absl.testing import flagsaver
import cloudpickle
from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats

from absl.testing import absltest


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
  ││  MapWithIndexMapDataset(transform=_add_dummy_metadata @ .../python/dataset/stats_test.py:129)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  MapMapDataset(transform=_identity @ .../python/dataset/stats_test.py:133)
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
  ││  MapIterDataset(transform=<lambda> @ .../python/dataset/stats_test.py:345)
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

_MIX_DATASET_REPR = r"""WARNING: Detected multi-parent datasets. Only displaying the first parent.

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


def _add_dummy_metadata(i, x):
  return {"data": x, "index": i, "epoch": 4, "dataset_index": 1}


def _identity(x):
  return x


class _AddOne(transforms.MapTransform):

  def map(self, x):
    return x + 1


def _make_stats_tree(cls):
  return cls(
      "root",
      [
          cls(
              "left",
              [
                  cls("left_left", []),
                  cls("left_right", []),
              ],
          ),
          cls(
              "right",
              [
                  cls("right_left", []),
                  cls("right_right", []),
              ],
          ),
      ],
  )


def _for_each_node(fn, nodes):
  to_visit = list(nodes)
  while to_visit:
    node = to_visit.pop(0)
    fn(node)
    to_visit.extend(node._parents)


class TimerTest(absltest.TestCase):

  def test_basic(self):
    timer = stats.Timer()
    self.assertEqual(timer.value(), 0)
    with timer:
      time.sleep(0.01)
    self.assertGreaterEqual(timer.value(), 0.01)
    with timer:
      time.sleep(0.02)
    self.assertGreaterEqual(timer.value(), 0.02)
    timer.reset()
    self.assertEqual(timer.value(), 0)


class NoopStatsTest(absltest.TestCase):

  def test_assert_is_noop(self):
    s = _make_stats_tree(stats.make_stats)
    self.assertIsInstance(s, stats._NoopStats)

  def test_record_self_time(self):
    s = _make_stats_tree(stats.make_stats)
    with s.record_self_time():
      pass
    s = s._parents[0]
    with s.record_self_time():
      pass

  def test_record_output_spec(self):
    s = _make_stats_tree(stats.make_stats)
    s.record_output_spec(1)
    s = s._parents[0]
    s.record_output_spec(1)

  def test_report(self):
    s = _make_stats_tree(stats.make_stats)
    s.report()
    s = s._parents[0]
    s.report()

if __name__ == "__main__":
  absltest.main()
