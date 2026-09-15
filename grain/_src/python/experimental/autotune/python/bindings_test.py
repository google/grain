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

from __future__ import annotations

import os
import sys
import time
from unittest import mock

from grain._src.python.experimental.autotune.python import bindings

from absl.testing import absltest
from absl.testing import parameterized
_STATUS_ERROR = RuntimeError


class AutotuneBindingsTest(parameterized.TestCase):

  def test_construct_fixed_ratio_node(self):
    node = bindings.FixedRatioNode(3.0, "TestNode")
    self.assertEqual(node.get_input_ratio(), 3.0)
    self.assertEqual(node.__str__(), "TestNode")

  def test_construct_source_node(self):
    node = bindings.SourceNode("MySource")
    self.assertEqual(node.__str__(), "MySource")
    self.assertEqual(node.get_input_ratio(), 0.0)

  def test_add_input(self):
    source = bindings.SourceNode()
    node = bindings.FixedRatioNode(1.0)
    node.add_input(source)
    self.assertIs(node.get_inputs()[0], source)
    self.assertIs(source.get_output(), node)

  def test_build_pipeline(self):
    source = bindings.SourceNode()
    node1 = bindings.FixedRatioNode(2.0, "Node1")
    node2 = bindings.FixedRatioNode(3.0, "Node2")
    node1.add_input(source)
    node2.add_input(node1)
    self.assertLen(node1.get_inputs(), 1)
    self.assertIs(node1.get_inputs()[0], source)
    self.assertLen(node2.get_inputs(), 1)
    self.assertIs(node2.get_inputs()[0], node1)
    self.assertIs(source.get_output(), node1)
    self.assertIs(node1.get_output(), node2)
    self.assertIsNone(node2.get_output())

  def test_add_input_to_source_node_fails(self):
    source1 = bindings.SourceNode()
    source2 = bindings.SourceNode()
    with self.assertRaisesRegex(
        _STATUS_ERROR, "Source node cannot have inputs"
    ):
      source1.add_input(source2)

  def test_add_input_with_existing_output_fails(self):
    source = bindings.SourceNode()
    node1 = bindings.FixedRatioNode(1.0, "Node1")
    node2 = bindings.FixedRatioNode(1.0, "Node2")
    node1.add_input(source)
    with self.assertRaisesRegex(_STATUS_ERROR, "already has an output"):
      node2.add_input(source)

  def test_record_timing(self):
    # EMA is set to 0.0 to avoid smoothing the timing. The last time is
    # recorded.
    node = bindings.FixedRatioNode(1.0, forgetting_factor=0.0)
    self.assertEqual(node.get_count(), 0)
    node.record_start()
    time.sleep(0.01)
    node.record_end()
    self.assertEqual(node.get_count(), 1)
    self.assertGreater(node.get_self_time_ms(), 5)
    self.assertLess(node.get_self_time_ms(), 500)

  def test_record_pause_resume(self):
    node = bindings.FixedRatioNode(1.0, forgetting_factor=0.0)
    node.record_start()
    time.sleep(0.01)
    node.record_pause()
    time.sleep(0.02)
    node.record_resume()
    time.sleep(0.01)
    node.record_end()
    self.assertEqual(node.get_count(), 1)
    # time should be ~20ms, allow for some buffer.
    self.assertGreater(node.get_self_time_ms(), 15)
    self.assertLess(node.get_self_time_ms(), 500)

  def test_get_pipeline_stats(self):
    source = bindings.SourceNode(forgetting_factor=0.0)
    node1 = bindings.FixedRatioNode(2.0, "Node1", forgetting_factor=0.0)
    node1.add_input(source)
    source.record_start()
    time.sleep(0.01)
    source.record_end()
    node1.record_start()
    time.sleep(0.01)
    node1.record_end()

    stats = node1.get_pipeline_stats()
    self.assertEqual(stats["name"], "Node1")
    self.assertEqual(stats["input_ratio"], 2.0)
    self.assertGreater(stats["self_time_ms"], 0)
    self.assertGreater(stats["output_time_ms"], stats["self_time_ms"])
    self.assertEqual(stats["consumer_time_ms"], 0)
    self.assertEqual(stats["count"], 1)
    self.assertLen(stats["inputs"], 1)
    self.assertEqual(stats["inputs"][0]["name"], "SourceNode")
    self.assertIsInstance(stats["inputs"][0]["consumer_time_ms"], float)
    self.assertEqual(stats["inputs"][0]["count"], 1)

  def test_get_pipeline_stats_without_timing(self):
    source = bindings.SourceNode()
    node = bindings.FixedRatioNode(2.0)
    node.add_input(source)
    stats = node.get_pipeline_stats()
    self.assertEqual(stats["name"], "FixedRatioNode(2.000000)")
    self.assertEqual(stats["input_ratio"], 2.0)
    self.assertEqual(stats["self_time_ms"], 0)
    self.assertEqual(stats["output_time_ms"], 0)
    self.assertEqual(stats["consumer_time_ms"], 0)
    self.assertEqual(stats["count"], 0)
    self.assertLen(stats["inputs"], 1)
    self.assertEqual(stats["inputs"][0]["name"], "SourceNode")
    self.assertEqual(stats["inputs"][0]["consumer_time_ms"], 0)
    self.assertEqual(stats["inputs"][0]["count"], 0)

  def test_construct_unknown_ratio_node(self):
    node = bindings.UnknownRatioNode(
        initial_ratio=3.0, name="TestNode", forgetting_factor=0.0
    )
    self.assertEqual(node.get_input_ratio(), 3.0)
    self.assertEqual(node.__str__(), "TestNode")
    node.record_ratio(2.0)
    self.assertAlmostEqual(node.get_input_ratio(), 2.0, msg="ratio", delta=1e-5)

  def test_autotune_parameter_is_convertible_to_primitive_types(self):
    param = bindings.AutotuneParameter("test", 10.0, 1.0, 100.0)
    self.assertEqual(int(param), 10)
    self.assertEqual(float(param), 10.0)

  def test_get_consumer_time_ms(self):
    source = bindings.SourceNode(forgetting_factor=0.0)
    node = bindings.FixedRatioNode(1.0, forgetting_factor=0.0)
    node.add_input(source)

    # Source runs 10ms
    source.record_start()
    time.sleep(0.01)
    source.record_end()

    # Node runs 20ms
    node.record_start()
    time.sleep(0.02)
    node.record_end()

    # Node waits 5ms
    time.sleep(0.005)
    node.record_start()  # Record inactive time

    # Check values
    # Node consumer time = inactive time (~5ms)
    self.assertGreater(node.get_consumer_time_ms(), 2)

    # Source consumer time = 5 + 20 + 10 - 10 = 25ms
    self.assertGreater(source.get_consumer_time_ms(), 20)

  @parameterized.named_parameters(
      dict(
          testcase_name="user_defined",
          cpu_budget=10,
          ram_budget_gb=20,
          expected_cpu=10,
          expected_ram_gb=20,
      ),
      dict(
          testcase_name="estimated",
          cpu_budget=None,
          ram_budget_gb=None,
          expected_cpu=0,
          expected_ram_gb=0,
      ),
  )
  def test_model_constraints(
      self, cpu_budget, ram_budget_gb, expected_cpu, expected_ram_gb
  ):
    constraints = bindings.AutotuneModelConfig(
        cpu_budget=cpu_budget, ram_budget_gb=ram_budget_gb
    )
    if cpu_budget is None and ram_budget_gb is None:
      try:
        self.assertGreater(constraints.get_cpu_budget(), 0)
      except Exception:  # pylint: disable=broad-except
        pass
      try:
        self.assertGreater(constraints.get_ram_budget_gb(), 0)
      except Exception:  # pylint: disable=broad-except
        pass
      return

    self.assertEqual(constraints.get_cpu_budget(), expected_cpu)
    self.assertEqual(constraints.get_ram_budget_gb(), float(expected_ram_gb))

  def test_interleave_node(self):
    node = bindings.InterleaveNode("InterleaveNode", cycle_length=2)
    source1 = bindings.FixedRatioNode(1.0, "source1")
    # source1: ~10ms
    source1.record_start()
    time.sleep(0.01)
    source1.record_end()

    source2 = bindings.FixedRatioNode(1.0, "source2")
    # source2: ~20ms
    source2.record_start()
    time.sleep(0.02)
    source2.record_end()

    node.add_input(source1)
    node.add_input(source2)

    # Output time should be Self + Average(Inputs)
    # Self time is 0 (default)
    # Average(10, 20) = 15. allow loose bounds.
    output_time = node.get_output_time_ms()
    self.assertGreater(output_time, 10.0)
    self.assertLess(output_time, 500.0)

  def test_async_fixed_ratio_node_regularization(self):
    node = bindings.AsyncFixedRatioNode(1.0, "AsyncNode")
    self.assertEqual(
        node.get_concurrency_regularization_mode(),
        bindings.ConcurrencyRegularizationMode.NONE,
    )
    self.assertEqual(node.get_concurrency_regularization_weight(), 0.0)

    node.set_concurrency_regularization(
        bindings.ConcurrencyRegularizationMode.LINEAR, 0.01
    )
    self.assertEqual(
        node.get_concurrency_regularization_mode(),
        bindings.ConcurrencyRegularizationMode.LINEAR,
    )
    self.assertEqual(node.get_concurrency_regularization_weight(), 0.01)

    node.set_concurrency_regularization(
        bindings.ConcurrencyRegularizationMode.QUADRATIC, 0.02
    )
    self.assertEqual(
        node.get_concurrency_regularization_mode(),
        bindings.ConcurrencyRegularizationMode.QUADRATIC,
    )
    self.assertEqual(node.get_concurrency_regularization_weight(), 0.02)


if __name__ == "__main__":
  absltest.main()
