# Copyright 2026 Google LLC
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

"""Tests for Autotune serialization."""

from __future__ import annotations

import multiprocessing as mp
from grain._src.python.experimental.autotune.python import bindings as autotune_bindings

from absl.testing import absltest


class AutotuneSerializationTest(absltest.TestCase):

  def test_source_node_snapshot(self):
    node = autotune_bindings.SourceNode("Source", 0.99)
    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertIsInstance(recovered_node, autotune_bindings.SourceNode)
    self.assertEqual(str(recovered_node), "Source")
    self.assertEqual(recovered_node.get_input_ratio(), 0.0)

  def test_fixed_ratio_node_snapshot(self):
    node = autotune_bindings.FixedRatioNode(2.0, "Fixed", 0.95)
    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertIsInstance(recovered_node, autotune_bindings.FixedRatioNode)
    self.assertEqual(str(recovered_node), "Fixed")
    self.assertEqual(recovered_node.get_input_ratio(), 2.0)

  def test_unknown_ratio_node_snapshot(self):
    node = autotune_bindings.UnknownRatioNode(1.5, "Unknown", 0.9)
    # Simulate some updates
    node.record_ratio(2.0)

    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertIsInstance(recovered_node, autotune_bindings.UnknownRatioNode)
    self.assertEqual(str(recovered_node), "Unknown")
    # The ratio should be close to the updated value
    self.assertAlmostEqual(recovered_node.get_input_ratio(), 2.0, delta=0.1)

  def test_async_fixed_ratio_node_snapshot(self):
    node = autotune_bindings.AsyncFixedRatioNode(
        input_ratio=1.0,
        name="Async",
        forgetting_factor=0.8,
        concurrency=4.0,
        buffer_size=10.0,
    )
    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertIsInstance(recovered_node, autotune_bindings.AsyncFixedRatioNode)
    self.assertEqual(str(recovered_node), "Async")
    self.assertTrue(recovered_node.is_async)
    self.assertEqual(recovered_node.concurrency, 4.0)
    self.assertEqual(recovered_node.buffer_size, 10.0)

  def test_interleave_node_snapshot(self):
    node = autotune_bindings.InterleaveNode("Interleave", cycle_length=3)
    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertIsInstance(recovered_node, autotune_bindings.InterleaveNode)
    self.assertEqual(str(recovered_node), "Interleave")
    self.assertAlmostEqual(
        recovered_node.get_input_ratio(), 1.0 / 3.0, places=5
    )

  def test_graph_snapshot(self):
    source = autotune_bindings.SourceNode("Source")
    mid = autotune_bindings.FixedRatioNode(2.0, "Mid")
    root = autotune_bindings.AsyncFixedRatioNode(1.0, "Root", 0.99, 2.0, 5.0)

    # Build graph: Root -> Mid -> Source
    mid.add_input(source)
    root.add_input(mid)

    snapshot_bytes = root.get_snapshot_bytes()
    recovered_root = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertEqual(str(recovered_root), "Root")
    inputs = recovered_root.get_inputs()
    self.assertLen(inputs, 1)

    recovered_mid = inputs[0]
    self.assertEqual(str(recovered_mid), "Mid")
    self.assertIsInstance(recovered_mid, autotune_bindings.FixedRatioNode)

    mid_inputs = recovered_mid.get_inputs()
    self.assertLen(mid_inputs, 1)

    recovered_source = mid_inputs[0]
    self.assertEqual(str(recovered_source), "Source")
    self.assertIsInstance(recovered_source, autotune_bindings.SourceNode)

  def test_stats_recovery(self):
    node = autotune_bindings.FixedRatioNode(1.0, "StatsNode")
    # Record some stats
    for _ in range(10):
      node.record_start()
      node.record_end()

    self.assertEqual(node.get_count(), 10)

    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertEqual(recovered_node.get_count(), 10)

  def test_concurrency_stats_recovery(self):
    node = autotune_bindings.FixedRatioNode(1.0, "ConcurrencyStatsNode")
    # Record some concurrency observations
    node.test_record_concurrency_observation(1.0, 10.0)
    node.test_record_concurrency_observation(2.0, 5.0)
    node.test_record_concurrency_observation(4.0, 3.0)

    self.assertEqual(node.get_concurrency_bins_seen(), 3)

    snapshot_bytes = node.get_snapshot_bytes()
    recovered_node = autotune_bindings.recover_autotune_node(snapshot_bytes)

    self.assertEqual(recovered_node.get_concurrency_bins_seen(), 3)
    node_stats = node.get_pipeline_stats()
    recovered_stats = recovered_node.get_pipeline_stats()

    self.assertAlmostEqual(
        recovered_stats["base_throughput"], node_stats["base_throughput"]
    )
    self.assertAlmostEqual(
        recovered_stats["contention"], node_stats["contention"]
    )
    self.assertAlmostEqual(
        recovered_stats["coherency"], node_stats["coherency"]
    )


if __name__ == "__main__":
  absltest.main()
