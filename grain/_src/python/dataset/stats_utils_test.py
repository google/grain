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

from grain._src.python import shared_memory_array
from grain._src.python.dataset import stats_utils
from grain.proto import execution_summary_pb2
import numpy as np

from absl.testing import absltest


class StatsUtilsTest(absltest.TestCase):

  def test_merge_execution_summaries(self):
    from_s = execution_summary_pb2.ExecutionSummary()
    to_s = execution_summary_pb2.ExecutionSummary()
    from_s.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapMapDataset",
            inputs=[1],
            total_processing_time_ns=4000,
            min_processing_time_ns=4,
            max_processing_time_ns=40,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
        )
    )
    to_s.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapMapDataset",
            inputs=[1],
            total_processing_time_ns=4000,
            min_processing_time_ns=40,
            max_processing_time_ns=400,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
        )
    )
    stats_utils.merge_execution_summaries(from_s, to_s)
    self.assertEqual(to_s.nodes[0].total_processing_time_ns, 8000)
    self.assertEqual(to_s.nodes[0].min_processing_time_ns, 4)
    self.assertEqual(to_s.nodes[0].max_processing_time_ns, 400)
    self.assertEqual(to_s.nodes[0].num_produced_elements, 20)

  def test_stats_summary_helper_class(self):
    main_summary = execution_summary_pb2.ExecutionSummary()
    workers_summary = execution_summary_pb2.ExecutionSummary()
    main_summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapDatasetIterator",
            inputs=[],
            total_processing_time_ns=4000,
            min_processing_time_ns=4,
            max_processing_time_ns=40,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
            is_output=True,
        )
    )
    workers_summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0,
            name="MapMapDataset",
            inputs=[1],
            total_processing_time_ns=4000,
            min_processing_time_ns=40,
            max_processing_time_ns=400,
            num_produced_elements=10,
            output_spec="<class 'int'>[]",
            is_output=True,
            bytes_produced=100,
        )
    )
    workers_summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=1,
            name="MapMapDataset",
            inputs=[],
            total_processing_time_ns=4000,
            min_processing_time_ns=400,
            max_processing_time_ns=4000,
            output_spec="<class 'int'>[]",
        )
    )
    complete_summary = stats_utils.get_complete_summary(
        main_summary, workers_summary, 1
    )
    self.assertLen(complete_summary.nodes, 3)
    self.assertEqual(complete_summary.nodes[0].inputs, [1])
    self.assertEqual(complete_summary.nodes[0].inputs, [1])
    self.assertEqual(complete_summary.nodes[1].inputs, [2])
    self.assertEqual(complete_summary.nodes[2].inputs, [])
    # The output node in the workers summary is the input to the root node in
    # the main summary.
    self.assertEqual(complete_summary.nodes[1].is_output, False)
    self.assertEqual(complete_summary.nodes[0].bytes_consumed, 100)

  def test_get_allocated_bytes(self):
    # Create elements of different types
    np_array = np.zeros(shape=(5, 2), dtype=np.int32)  # 5*2*4 = 40 bytes
    test_string = "hello world"  # 11 bytes
    shm_meta = shared_memory_array.SharedMemoryArrayMetadata(
        name="test_key", dtype=np.float32, shape=(10,)
    )  # 10*4 = 40 bytes
    self.assertEqual(stats_utils.calculate_allocated_bytes(np_array), 40)
    self.assertEqual(stats_utils.calculate_allocated_bytes(test_string), 11)
    self.assertEqual(stats_utils.calculate_allocated_bytes(shm_meta), 40)

  def test_sort_nodes_by_wait_time_ratio(self):
    summary = execution_summary_pb2.ExecutionSummary()
    summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(id=0, wait_time_ratio=0.8)
    )
    summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(id=1, wait_time_ratio=0.2)
    )
    summary.nodes[2].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(id=2, wait_time_ratio=0.5)
    )

    sorted_summary = stats_utils.sort_nodes_by_wait_time_ratio(summary)

    # Check if the nodes are sorted by wait_time_ratio and re-keyed
    self.assertLen(sorted_summary.nodes, 3)
    self.assertEqual(sorted_summary.nodes[0].wait_time_ratio, 0.2)
    self.assertEqual(sorted_summary.nodes[0].id, 1)  # Original ID
    self.assertEqual(sorted_summary.nodes[1].wait_time_ratio, 0.5)
    self.assertEqual(sorted_summary.nodes[1].id, 2)  # Original ID
    self.assertEqual(sorted_summary.nodes[2].wait_time_ratio, 0.8)
    self.assertEqual(sorted_summary.nodes[2].id, 0)  # Original ID

  def test_analyze_summary_source_bottleneck(self):
    summary = execution_summary_pb2.ExecutionSummary()
    summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(id=0, name="MapDataset")
    )
    # top node
    summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=1, name="SourceMapDataset"
        )
    )
    detected_bottleneck = stats_utils.analyze_summary(summary)
    self.assertLen(detected_bottleneck, 1)
    self.assertIn(
        "Your source is likely the bottleneck",
        detected_bottleneck[0],
    )

  def test_analyze_summary_no_source_bottleneck(self):
    summary = execution_summary_pb2.ExecutionSummary()
    summary.nodes[0].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(
            id=0, name="SourceMapDataset"
        )
    )
    # top node
    summary.nodes[1].CopyFrom(
        execution_summary_pb2.ExecutionSummary.Node(id=1, name="MapDataset")
    )
    detected_bottleneck = stats_utils.analyze_summary(summary)
    self.assertEmpty(detected_bottleneck)


if __name__ == "__main__":
  absltest.main()
