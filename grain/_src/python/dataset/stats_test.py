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
import contextlib
import functools
import re
import sys
import threading
import time
from unittest import mock
import weakref

from absl import flags
from absl.testing import flagsaver
import cloudpickle
from grain._src.core import transforms
from grain._src.python import options
from grain._src.python import shared_memory_array
from grain._src.python.dataset import dataset
from grain._src.python.dataset import stats
from grain.proto import execution_summary_pb2
import numpy as np

from absl.testing import absltest


_MAP_DATASET_REPR = r"""RangeMapDataset(start=0, stop=10, step=1)
  ││
  ││  
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  WithOptionsMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  ShuffleMapDataset
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  SliceMapDataset[1:10:3]
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapWithIndexMapDataset(transform=_add_dummy_metadata @ .../python/dataset/stats_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  MapMapDataset(transform=_identity @ .../python/dataset/stats_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  RepeatMapDataset(num_epochs=2)
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
  ││  PrefetchDatasetIterator(read_options=ReadOptions(num_threads=16, prefetch_buffer_size=500), allow_nones=False)
  ││
  ╲╱
"<class 'int'>[]"

  ││
  ││  MapDatasetIterator(transform=<lambda> @ .../python/dataset/stats_test.py:XXX)
  ││
  ╲╱
{'data': "<class 'int'>[]",
 'dataset_index': "<class 'int'>[]",
 'epoch': "<class 'int'>[]",
 'index': "<class 'int'>[]"}

  ││
  ││  BatchDatasetIterator(batch_size=2, drop_remainder=False)
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
      stats.StatsConfig(name="root"),
      [
          cls(
              stats.StatsConfig(name="left"),
              [
                  cls(stats.StatsConfig(name="left_left"), []),
                  cls(stats.StatsConfig(name="left_right"), []),
              ],
          ),
          cls(
              stats.StatsConfig(name="right"),
              [
                  cls(stats.StatsConfig(name="right_left"), []),
                  cls(stats.StatsConfig(name="right_right"), []),
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


@contextlib.contextmanager
def _unparse_flags():
  argv = sys.argv
  flags.FLAGS.unparse_flags()
  try:
    yield
  finally:
    flags.FLAGS(argv)


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


class DefaultStatsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    stats._iter_weakref_registry.clear()

  def test_assert_is_default(self):
    s = _make_stats_tree(stats.make_stats)
    self.assertIsInstance(s, stats._DefaultStats)

  def test_with_flags_not_parsed(self):
    # unparse the debug flags explicitly
    with _unparse_flags():
      s = stats.make_stats(stats.StatsConfig(name="test_stats"), ())
      self.assertIsInstance(s, stats._DefaultStats)

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


class DebugModeStatsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(flagsaver.flagsaver(grain_py_debug_mode=True))
    stats._iter_weakref_registry.clear()

  @mock.patch.object(stats, "_REPORTING_PERIOD_SEC", 0.05)
  def test_record_stats(self):
    s = _make_stats_tree(stats.make_stats)
    self.assertIsInstance(s, stats._ExecutionStats)
    flat_stats = []
    to_visit = [s]
    while to_visit:
      node = to_visit.pop(0)
      flat_stats.append(node)
      to_visit.extend(node._parents)

    reported_self_times = collections.defaultdict(int)

    def mock_report(node):
      while node._self_times_buffer:
        reported_self_times[id(node)] += node._self_times_buffer.pop()
      for p in node._parents:
        p.report()

    for node in flat_stats:
      node.report = functools.partial(mock_report, node)
    for node in flat_stats:
      with node.record_self_time(offset_ns=10**9):
        time.sleep(0.5)
    time.sleep(1)
    self_times = list(reported_self_times.values())
    self.assertLen(self_times, len(flat_stats))
    for self_time in self_times:
      self.assertGreaterEqual(self_time, 1.05 * 10**9)

  @mock.patch.object(stats, "_REPORTING_PERIOD_SEC", 0.05)
  def test_record_stats_thread_safe(self):
    s = stats.make_stats(stats.StatsConfig(name="test_stats"), ())
    reported_self_time = 0

    def mock_report(node):
      while node._self_times_buffer:
        nonlocal reported_self_time
        reported_self_time += node._self_times_buffer.pop()
      for p in node._parents:
        p.report()

    s.report = functools.partial(mock_report, s)

    def record_self_time():
      with s.record_self_time():
        # Sleep releases GIL, so this will actually execute concurrently.
        time.sleep(1)

    n_threads = 100
    recording_threads = []
    for _ in range(n_threads):
      t = threading.Thread(target=record_self_time)
      t.start()
      recording_threads.append(t)
    for t in recording_threads:
      t.join()
    time.sleep(1)
    self.assertGreaterEqual(reported_self_time, n_threads)

  def test_picklable(self):
    mock_iter = mock.Mock()
    s = stats.make_stats(
        stats.StatsConfig(
            name="test_stats", iter_weakref=weakref.ref(mock_iter)
        ),
        (),
    )
    self.assertIsInstance(s, stats._ExecutionStats)
    s = cloudpickle.loads(cloudpickle.dumps(s))
    self.assertIsInstance(s, stats._ExecutionStats)
    with s.record_self_time():
      time.sleep(0.5)
    s = cloudpickle.loads(cloudpickle.dumps(s))
    self.assertIsInstance(s, stats._ExecutionStats)

  def test_pretty_print_execution_summary(self):
    dummy_summary = execution_summary_pb2.ExecutionSummary()
    dummy_summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapDatasetIterator(transform=_MapFnFromPreprocessingBuilder(preprocessing_builder=NextTokenAsTargetTextPreprocessingBuilder))",
            inputs=[1],
            wait_time_ratio=0.5,
            total_processing_time_ns=0,
            min_processing_time_ns=400_000,
            max_processing_time_ns=0,
            num_produced_elements=0,
            output_spec="<class 'int'>[]",
        )
    )
    dummy_summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=1,
            name="PrefetchDatasetIterator",
            inputs=[2],
            wait_time_ratio=0.5,
            total_processing_time_ns=400_000,
            min_processing_time_ns=400,
            max_processing_time_ns=40000,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
            is_output=True,
            is_prefetch=True,
            bytes_consumed=100000,
            bytes_produced=50000,
        )
    )
    dummy_summary.nodes[2].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=2,
            name="MapMapDataset",
            inputs=[3, 4],
            wait_time_ratio=0.375,
            total_processing_time_ns=400_000_000,
            min_processing_time_ns=4000,
            max_processing_time_ns=40_000_000,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
        )
    )
    dummy_summary.nodes[3].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=3,
            name="RangeMapDataset",
            wait_time_ratio=0.125,
            total_processing_time_ns=4000_000_000,
            min_processing_time_ns=400_000,
            max_processing_time_ns=400_000_000,
            num_produced_elements=10,
            inputs=[],
            output_spec="<class 'int'>[]",
        )
    )
    dummy_summary.nodes[4].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=4,
            name="RangeMapDataset",
            total_processing_time_ns=0,
            wait_time_ratio=0,
            min_processing_time_ns=400_000,
            max_processing_time_ns=0,
            num_produced_elements=0,
            inputs=[],
            output_spec="<class 'int'>[]",
        )
    )

    expected_result = """
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id | name                           | inputs | percent wait time | total processing time | min processing time | max processing time | avg processing time | num produced elements | memory usage |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4  | RangeMapDataset                | []     | 0.00%             | N/A                   | N/A                 | N/A                 | N/A                 | 0                     | 0 bytes      |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3  | RangeMapDataset                | []     | 12.50%            | 4.00s                 | 400.00us            | 400.00ms            | 400.00ms            | 10                    | 0 bytes      |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2  | MapMapDataset                  | [3, 4] | 37.50%            | 400.00ms              | 4.00us              | 40.00ms             | 40.00ms             | 10                    | 0 bytes      |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | PrefetchDatasetIterator        | [2]    | N/A               | 400.00us              | 400ns               | 40.00us             | 40.00us             | 10                    | 48.83 KiB    |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0  | MapDatasetIterator(transform=_ | [1]    | 50.00%            | N/A                   | N/A                 | N/A                 | N/A                 | 0                     | 0 bytes      |
|    | MapFnFromPreprocessingBuilder( |        |                   |                       |                     |                     |                     |                       |              |
|    | preprocessing_builder=NextToke |        |                   |                       |                     |                     |                     |                       |              |
|    | nAsTargetTextPreprocessingBuil |        |                   |                       |                     |                     |                     |                       |              |
|    | der))                          |        |                   |                       |                     |                     |                     |                       |              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
"""
    self.assertEqual(
        expected_result,
        "\n" + stats.pretty_format_summary(dummy_summary),
    )

  def test_compute_iterator_wait_time_ratio(self):
    dummy_summary = execution_summary_pb2.ExecutionSummary()
    dummy_summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapDatasetIterator",
            inputs=[1],
            total_processing_time_ns=4000_000_000,
            min_processing_time_ns=400,
            max_processing_time_ns=40000,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
            is_output=True,
        )
    )
    dummy_summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=1,
            name="PrefetchDatasetIterator",
            inputs=[2],
            total_processing_time_ns=4000_000_000,
            min_processing_time_ns=4000,
            max_processing_time_ns=40_000_000,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
            is_prefetch=True,
        )
    )
    dummy_summary.nodes[2].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=2,
            name="MapMapDataset",
            inputs=[3],
            total_processing_time_ns=1000_000_000,
            min_processing_time_ns=400_000,
            max_processing_time_ns=400_000_000,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
        )
    )
    dummy_summary.nodes[3].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=3,
            name="RangeMapDataset",
            total_processing_time_ns=3000_000_000,
            min_processing_time_ns=400_000,
            max_processing_time_ns=4000_000,
            num_produced_elements=10,
            inputs=[],
            output_spec="<class 'int'>[]",
        )
    )
    stats._populate_wait_time_ratio(dummy_summary)
    self.assertEqual(dummy_summary.nodes[0].wait_time_ratio, 0.5)
    self.assertEqual(dummy_summary.nodes[1].wait_time_ratio, 0)
    self.assertEqual(dummy_summary.nodes[2].wait_time_ratio, 0.125)
    self.assertEqual(dummy_summary.nodes[3].wait_time_ratio, 0.375)

  @flagsaver.flagsaver(grain_py_dataset_visualization_output_dir="TEST_DIR")
  def test_dataset_visualization_with_output_dir(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "Saving the dataset graph to a file is not supported yet.",
    ):
      _ = list(ds)

  def test_memory_usage(self):
    # Create elements of different types
    np_array = np.zeros(shape=(5, 2), dtype=np.int32)  # 5*2*4 = 40 bytes
    test_string = "hello world"  # 11 bytes
    shm_meta = shared_memory_array.SharedMemoryArrayMetadata(
        name="test_key", dtype=np.float32, shape=(10,)
    )  # 10*4 = 40 bytes

    ds = (
        dataset.MapDataset.source([np_array, test_string, shm_meta])
        .map(lambda x: x)
        .to_iter_dataset(
            read_options=options.ReadOptions(prefetch_buffer_size=10)
        )
    )
    it = ds.__iter__()
    _ = list(it)
    local_stats = it._stats
    self.assertIsInstance(local_stats, stats._ExecutionStats)
    # Force aggregate the buffered stats.
    local_stats.report()
    self.assertEqual(local_stats._summary.bytes_consumed, 91)
    self.assertEqual(local_stats._summary.bytes_produced, 91)


class GraphModeStatsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        flagsaver.flagsaver(grain_py_dataset_visualization_output_dir="")
    )

  def _assert_visualization(self, ds, expected):
    result = ds._stats._visualize_dataset_graph()  # pytype: disable=attribute-error
    # Remove line number from the result to make test less brittle.
    result = re.sub(r".py:\d+", ".py:XXX", result)
    self.assertEqual(result, expected)

  @flagsaver.flagsaver(grain_py_debug_mode=True)
  def test_visualization_in_debug_mode(self):
    ds = (
        dataset.MapDataset.range(10)
        .seed(42)
        .shuffle()
        .slice(slice(1, None, 3))
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
        .repeat(2)
    )
    # Visualization graph is constructed while iterating through pipeline.
    _ = list(ds)
    self.assertIsInstance(ds._stats, stats._ExecutionStats)
    self._assert_visualization(ds, _MAP_DATASET_REPR)

  def test_visualize_map(self):
    ds = (
        dataset.MapDataset.range(10)
        .seed(42)
        .shuffle()
        .slice(slice(1, None, 3))
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
        .repeat(2)
    )
    # Visualization graph is constructed while iterating through pipeline.
    _ = list(ds)
    self.assertIsInstance(ds._stats, stats._VisualizationStats)
    self._assert_visualization(ds, _MAP_DATASET_REPR)

  def test_visualize_iter(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .to_iter_dataset()
        .seed(42)
        .map(lambda x: _add_dummy_metadata(2, x))
        .batch(2)
    )
    # Visualization graph is constructed while iterating through pipeline.
    it = ds.__iter__()
    _ = list(it)
    self.assertIsInstance(it._stats, stats._VisualizationStats)
    self._assert_visualization(it, _ITER_DATASET_REPR)

  def test_visualize_with_mix(self):
    ds1 = dataset.MapDataset.range(10).shuffle(42)
    ds2 = dataset.MapDataset.range(10).shuffle(43)
    ds = dataset.MapDataset.mix([ds1, ds2]).map(_AddOne())
    # Visualization graph is constructed while iterating through pipeline.
    _ = list(ds)
    self.assertIsInstance(ds._stats, stats._VisualizationStats)
    self._assert_visualization(ds, _MIX_DATASET_REPR)

  @flagsaver.flagsaver(grain_py_dataset_visualization_output_dir="TEST_DIR")
  def test_dataset_visualization_with_output_dir(self):
    ds = (
        dataset.MapDataset.range(10)
        .shuffle(42)
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
    )
    with self.assertRaisesRegex(
        NotImplementedError,
        "Saving the dataset graph to a file is not supported yet.",
    ):
      _ = list(ds)

  def test_picklable(self):
    ds = (
        dataset.MapDataset.range(10)
        .seed(42)
        .shuffle()
        .slice(slice(1, None, 3))
        .map_with_index(_add_dummy_metadata)
        .map(_identity)
        .repeat(2)
    )
    ds = cloudpickle.loads(cloudpickle.dumps(ds))
    # Visualization graph is constructed while iterating through pipeline.
    _ = list(ds)
    self.assertIsInstance(ds._stats, stats._VisualizationStats)
    self._assert_visualization(ds, _MAP_DATASET_REPR)

  @flagsaver.flagsaver(grain_py_dataset_visualization_output_dir=None)
  def test_dataset_visualization_with_output_dir_none(self):
    s = stats.make_stats(stats.StatsConfig(name="test_stats"), ())
    self.assertIsInstance(s, stats._DefaultStats)


if __name__ == "__main__":
  absltest.main()
