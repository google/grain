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

"""Autotune Python library tests."""

from __future__ import annotations

import pprint
import time
from unittest import mock

from absl.testing import absltest
import grain
import multiprocessing as mp
from grain._src.core import tree_lib
from grain._src.python import options
from grain._src.python.dataset.transformations import autotune
from grain._src.python.dataset.transformations import interleave
from grain._src.python.dataset.transformations import prefetch
from grain._src.python.experimental.autotune.python import bindings
import numpy as np


def _get_test_model_config():
  return bindings.AutotuneModelConfig(ram_budget_gb=4.0, cpu_budget=4)


class AutotuneTest(absltest.TestCase):

  def test_autotune_iter_dataset_simple(self):
    """Test that a simple dataset works when wrapped with autotune dataset iterators."""
    ds = grain.MapDataset.range(10).to_iter_dataset().map(lambda x: x + 1)
    ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=True
    )
    elements = list(ds)
    self.assertListEqual(elements, list(range(1, 11)))

  def test_autotune_iter_dataset_map_batch(self):
    ds = (
        grain.MapDataset.range(10)
        .map(lambda x: x)
        .to_iter_dataset()
        .map(lambda x: x + 1)
        .batch(2)
    )
    ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=False
    )
    elements = list(ds)
    np.testing.assert_equal(elements, [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    stats = ds._autotune_node.get_pipeline_stats()
    # 5 batches + stop.
    self.assertEqual(stats["count"], 6)
    self.assertEqual(stats["input_ratio"], 2.0)

    # Map Node
    map_node = stats["inputs"][0]
    self.assertEqual(map_node["input_ratio"], 1.0)
    # Prefetch Node
    prefetch_node = map_node["inputs"][0]
    self.assertEqual(prefetch_node["input_ratio"], 1.0)
    # Map Node
    map_node_2 = prefetch_node["inputs"][0]
    self.assertEqual(map_node_2["input_ratio"], 1.0)
    # Range (Source) Node
    range_node = map_node_2["inputs"][0]
    self.assertEqual(range_node["input_ratio"], 0.0)

  def test_timing_is_recorded(self):
    def slow_map(x):
      time.sleep(0.01)
      return x

    ds = grain.MapDataset.range(3).map(slow_map).to_iter_dataset()
    ds = ds.map(slow_map)
    ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=True
    )
    elements = list(ds)
    self.assertListEqual(elements, list(range(3)))
    stats = ds._autotune_node.get_pipeline_stats()
    # The map should take at least 5e-3 seconds to execute.
    self.assertGreater(stats["self_time_ms"], 5)
    # Check that the map before to_iter_dataset is also recorded.
    prefetch_node = stats["inputs"][0]
    map_node = prefetch_node["inputs"][0]
    self.assertGreater(map_node["self_time_ms"], 5)

  def test_element_size_bytes_is_recorded(self):
    def make_bytes(_):
      return b"a" * 100

    ds = grain.MapDataset.range(10).map(make_bytes).to_iter_dataset()
    ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=True
    )
    elements = list(ds)
    self.assertListEqual(elements, [b"a" * 100] * 10)
    # The element size should be 100 bytes.
    self.assertGreater(ds._autotune_node.get_element_size_bytes(), 90)
    self.assertLess(ds._autotune_node.get_element_size_bytes(), 110)
    self.assertEqual(ds._autotune_node.get_min_element_size_bytes(), 100)
    self.assertEqual(ds._autotune_node.get_max_element_size_bytes(), 100)

  def test_map_dataset_element_size_bytes_is_recorded(self):
    def make_bytes(_):
      return b"a" * 100

    ds = grain.MapDataset.range(10).map(make_bytes).to_iter_dataset()
    ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=True
    )
    elements = list(ds)
    self.assertListEqual(elements, [b"a" * 100] * 10)
    # The element size should be 100 bytes.
    self.assertGreater(ds._autotune_node.get_element_size_bytes(), 90)
    self.assertLess(ds._autotune_node.get_element_size_bytes(), 110)
    map_node = ds._autotune_node.get_inputs()[0]
    self.assertGreater(map_node.get_element_size_bytes(), 90)
    self.assertLess(map_node.get_element_size_bytes(), 110)
    self.assertEqual(map_node.get_min_element_size_bytes(), 100)
    self.assertEqual(map_node.get_max_element_size_bytes(), 100)

  def test_nodes_can_be_skipped(self):
    ds = grain.MapDataset.range(10).to_iter_dataset().map(lambda x: x + 1)
    ds_with_options = grain.experimental.WithOptionsIterDataset(
        ds, grain.experimental.DatasetOptions()
    )
    autotuned_ds = autotune.autotune(
        ds, model_config=_get_test_model_config(), allow_unknown_nodes=True
    )
    autotuned_ds_with_options = autotune.autotune(
        ds_with_options,
        model_config=_get_test_model_config(),
        allow_unknown_nodes=True,
    )
    # Required to build the model, since it requires inspecting the iterator
    # structure.
    autotuned_ds.__iter__()
    autotuned_ds_with_options.__iter__()
    # Check that the tree structure is the same
    tree_lib.assert_same_structure(
        autotuned_ds._autotune_node.get_pipeline_stats(),
        autotuned_ds_with_options._autotune_node.get_pipeline_stats(),
    )

  def test_prefetch_autotune_node_type(self):
    ds = grain.MapDataset.range(3)
    ds0 = ds.to_iter_dataset(
        options.ReadOptions(prefetch_buffer_size=0, num_threads=1)
    )
    ds1 = ds.to_iter_dataset(
        options.ReadOptions(prefetch_buffer_size=1, num_threads=1)
    )
    autotune_ds0 = autotune.autotune(ds0, model_config=_get_test_model_config())
    autotune_ds1 = autotune.autotune(ds1, model_config=_get_test_model_config())
    # Build models
    iter(autotune_ds0)
    iter(autotune_ds1)
    self.assertIsInstance(autotune_ds0._autotune_node, bindings.FixedRatioNode)
    self.assertIsInstance(
        autotune_ds1._autotune_node,
        bindings.AsyncFixedRatioNode,
    )

  def test_prefetch_parameters(self):
    """Verifies that PrefetchDatasetIterator exposes AutotuneParameters."""
    ds = grain.MapDataset.range(3)
    ds1 = ds.to_iter_dataset(
        options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=5, min_value=1, max_value=10
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=2, min_value=1, max_value=4
            ),
        )
    )
    # Autotune creates the node graph by traversing ds1.
    autotune_ds = autotune.autotune(ds1, model_config=_get_test_model_config())

    # Build the model by creating an iterator
    iter(autotune_ds)

    node = autotune_ds._autotune_node
    self.assertIsInstance(node, bindings.AsyncFixedRatioNode)
    self.assertEqual(node.buffer_size, 5.0)
    self.assertEqual(node.concurrency, 2.0)

    # Verify linkage by manually creating an iterator and node
    it = iter(ds1)
    self.assertIsNotNone(it.autotune_buffer_size)

    node2 = autotune._create_autotune_node(it)
    self.assertIsInstance(node2, bindings.AsyncFixedRatioNode)
    self.assertEqual(node2.buffer_size, 5.0)

    # Update parameter and check node
    it.autotune_buffer_size.set_value(10.0)
    node3 = autotune._create_autotune_node(it)
    self.assertEqual(node3.buffer_size, 10.0)

    it.autotune_num_threads.set_value(4.0)
    node4 = autotune._create_autotune_node(it)
    self.assertEqual(node4.concurrency, 4.0)

    # Call __next__ to trigger buffer size and concurrency updates
    next(it)
    self.assertEqual(it._target_prefetch_buffer_size, 10)
    self.assertEqual(it._target_num_threads, 4)

  def test_thread_prefetch_parameters(self):
    """Verifies AutotuneParameters exposed by ThreadPrefetchDatasetIterator."""
    ds = grain.MapDataset.range(3).to_iter_dataset()
    ds1 = prefetch.ThreadPrefetchIterDataset(
        ds,
        prefetch_buffer_size=bindings.AutotuneParameter(
            "buffer_size", initial_value=5, min_value=1, max_value=10
        ),
    )
    # Autotune creates the node graph by traversing ds1.
    autotune_ds = autotune.autotune(ds1, model_config=_get_test_model_config())

    # Build the model by creating an iterator
    iter(autotune_ds)

    node = autotune_ds._autotune_node
    self.assertIsInstance(node, bindings.AsyncFixedRatioNode)
    self.assertEqual(node.buffer_size, 5.0)

    # Verify linkage by manually creating an iterator and node
    it = iter(ds1)
    self.assertIsNotNone(it.autotune_buffer_size)

    node2 = autotune._create_autotune_node(it)
    self.assertIsInstance(node2, bindings.AsyncFixedRatioNode)
    self.assertEqual(node2.buffer_size, 5.0)

    # Update parameter and check node
    it.autotune_buffer_size.set_value(10.0)
    node3 = autotune._create_autotune_node(it)
    self.assertEqual(node3.buffer_size, 10.0)

    # Call __next__ to trigger buffer size update
    next(it)
    self.assertEqual(it._target_prefetch_buffer_size, 10)

  def test_interleave_stats(self):
    ds = grain.MapDataset.range(10)
    ds = ds.to_iter_dataset()
    ds = interleave.InterleaveIterDataset(datasets=[ds, ds], cycle_length=2)
    ds = autotune.autotune(ds, model_config=_get_test_model_config())
    ds_iter = ds.__iter__()
    next(ds_iter)
    interleave_node = ds._autotune_node.get_pipeline_stats()
    prefetch_node = interleave_node["inputs"][0]

    self.assertIn("Interleave", interleave_node["name"])
    self.assertIn("ThreadPrefetch", prefetch_node["name"])
    self.assertLen(interleave_node["inputs"], 1)

    next(ds_iter)
    interleave_node = ds._autotune_node.get_pipeline_stats()
    self.assertLen(interleave_node["inputs"], 2)
    prefetch_node1 = interleave_node["inputs"][0]
    prefetch_node2 = interleave_node["inputs"][1]
    self.assertIn("Interleave", interleave_node["name"])
    self.assertIn("ThreadPrefetch", prefetch_node1["name"])
    self.assertIn("ThreadPrefetch", prefetch_node2["name"])
    self.assertLen(interleave_node["inputs"], 2)

    prefetch_node1_output_time = prefetch_node1["output_time_ms"]
    prefetch_node2_output_time = prefetch_node2["output_time_ms"]
    interleave_output_time = interleave_node["output_time_ms"]
    interleave_self_time = interleave_node["self_time_ms"]
    self.assertAlmostEqual(
        interleave_output_time - interleave_self_time,
        (prefetch_node1_output_time + prefetch_node2_output_time) / 2.0,
        delta=1e-5,
    )

  def test_pipeline_stats_populated(self):
    def slow_map(x):
      time.sleep(0.01)
      return x

    ds = grain.MapDataset.range(3000).map(slow_map)
    ds = ds.to_iter_dataset(
        options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=2, min_value=1, max_value=100
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=4, min_value=1, max_value=100
            ),
        )
    ).batch(32)

    autotune_ds = autotune.autotune(ds, model_config=_get_test_model_config())
    it = iter(autotune_ds)

    for _ in range(55):
      next(it)

    stats = autotune_ds._autotune_node.get_pipeline_stats()
    self.assertIn("count", stats)
    self.assertGreater(stats["count"], 0)

    print("\n\nFinal Stats:")
    pprint.pprint(stats)

  def test_optimization_is_triggered_and_prefetch_picks_up_changes(self):
    def identity_map(x):
      return x

    ds = grain.MapDataset.range(10).map(identity_map)
    ds = ds.to_iter_dataset(
        options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=5, min_value=1, max_value=10
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=2, min_value=1, max_value=4
            ),
        )
    )
    config = bindings.AutotuneModelConfig(
        optimization_frequency=1, ram_budget_gb=4.0, cpu_budget=4
    )
    autotune_ds = autotune.autotune(
        ds, model_config=config, allow_unknown_nodes=True
    )

    it = iter(autotune_ds)
    # Replace the model with a Python mock
    mock_model = mock.create_autospec(bindings.AutotuneModel, instance=True)
    it._model = mock_model
    mock_model.maybe_optimize.return_value = True

    def mock_optimize_side_effect(_):
      it._inner.autotune_num_threads.set_value(3)
      it._inner.autotune_buffer_size.set_value(3)
      return True

    mock_model.maybe_optimize.side_effect = mock_optimize_side_effect

    # Initial target values.
    self.assertEqual(it._inner._target_num_threads, 2)
    self.assertEqual(it._inner._target_prefetch_buffer_size, 5)

    # Calling next(it) will:
    # 1. Call mock_model.maybe_optimize(), which executes side effect,
    #    updating parameter values to 3.
    # 3. Call PrefetchDatasetIterator.__next__(), which calls
    #    _apply_autotune_updates_if_present(), reading updated parameter values
    #    and updating _target_num_threads and _target_prefetch_buffer_size to 3.
    next(it)

    mock_model.maybe_optimize.assert_called_once_with(
        autotune_ds._autotune_node
    )
    self.assertEqual(it._inner.autotune_num_threads.get_value(), 3)
    self.assertEqual(it._inner.autotune_buffer_size.get_value(), 3)
    self.assertEqual(it._inner._target_num_threads, 3)
    self.assertEqual(it._inner._target_prefetch_buffer_size, 3)

  def test_prefetch_executor_rewrapping_on_resize(self):
    """Verifies that ThreadPoolExecutor is dynamically re-wrapped on resize."""
    ds = grain.MapDataset.range(10)
    ds1 = ds.to_iter_dataset(
        options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=5, min_value=1, max_value=10
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=2, min_value=1, max_value=4
            ),
        )
    )
    config = _get_test_model_config()
    autotune_ds = autotune.autotune(ds1, model_config=config)
    it = iter(autotune_ds)
    next(it)

    inner_it = it._inner
    self.assertIsInstance(inner_it._executor, autotune._AutotuneExecutorProxy)
    initial_executor = inner_it._executor

    # Dynamically update target thread count and trigger re-creation
    inner_it.autotune_num_threads.set_value(4.0)
    next(it)

    self.assertNotEqual(inner_it._executor, initial_executor)
    self.assertIsInstance(inner_it._executor, autotune._AutotuneExecutorProxy)
    self.assertEqual(inner_it._target_num_threads, 4)

  def test_warmup_state_is_respected(self):
    class CustomAutotuneModelConfig(bindings.AutotuneModelConfig):

      def __init__(self, warmup_steps: int):
        super().__init__(
            optimization_frequency=1,
            warmup_steps=warmup_steps,
            ram_budget_gb=4.0,
            cpu_budget=4,
        )

    ds = grain.MapDataset.range(10).to_iter_dataset()
    config = CustomAutotuneModelConfig(warmup_steps=3)

    autotune_ds = autotune.autotune(
        ds, model_config=config, allow_unknown_nodes=True
    )

    it = iter(autotune_ds)
    mock_model = mock.create_autospec(bindings.AutotuneModel, instance=True)
    it._model = mock_model
    mock_model.maybe_optimize.return_value = False

    next(it)
    next(it)
    next(it)
    # The first 3 steps should bypass optimization and timer entirely
    mock_model.maybe_optimize.assert_not_called()
    stats = autotune_ds._autotune_node.get_pipeline_stats()
    self.assertEqual(stats["count"], 0)

    next(it)
    # Step 4 should run optimization
    mock_model.maybe_optimize.assert_called_once()
    stats = autotune_ds._autotune_node.get_pipeline_stats()
    self.assertEqual(stats["count"], 1)

  def test_process_prefetch_stats_propagation(self):
    """Verifies that stats are propagated from worker to main process."""
    ds = grain.MapDataset.range(10000)
    ds = ds.to_iter_dataset()
    ds = ds.mp_prefetch(options=options.MultiprocessingOptions(num_workers=1))
    ds = autotune.autotune(
        ds,
        model_config=bindings.AutotuneModelConfig(
            ram_budget_gb=4.0, cpu_budget=4
        ),
        allow_unknown_nodes=True,
    )
    ds_iter = ds.__iter__()

    # Consume elements to trigger stats reporting and reception.
    for _ in range(110):
      next(ds_iter)
    while True:
      # Wait for worker stats to be reported.
      stats = ds_iter._autotune_node.get_pipeline_stats()
      node3 = stats["inputs"][0]["inputs"][0]
      if node3["inputs"] and node3["inputs"][0]["count"] > 0:
        break
      time.sleep(0.05)
      next(ds_iter)

    node1 = stats
    node2 = node1["inputs"][0]
    node3 = node2["inputs"][0]
    node4 = node3["inputs"][0]
    node5 = node4["inputs"][0]
    self.assertIn("Interleave", node1["name"])
    self.assertIn("ThreadPrefetch", node2["name"])
    self.assertIn("ProcessPrefetch", node3["name"])
    self.assertIn("Prefetch", node4["name"])
    self.assertIn("Range", node5["name"])

  def test_multiprocess_stats(self):
    """Verifies multiprocess stats propagation and merging."""
    ds = grain.MapDataset.range(10000)
    ds = ds.to_iter_dataset()
    ds = ds.mp_prefetch(options=options.MultiprocessingOptions(num_workers=3))
    ds = autotune.autotune(
        ds,
        model_config=bindings.AutotuneModelConfig(
            ram_budget_gb=4.0, cpu_budget=4
        ),
        allow_unknown_nodes=True,
    )
    ds_iter = ds.__iter__()

    # Consume elements to trigger stats reporting and reception.
    for _ in range(350):
      next(ds_iter)
    while True:
      # Wait for worker stats to be reported.
      stats = ds_iter._autotune_node.get_pipeline_stats()
      node3 = stats["inputs"][0]["inputs"][0]
      if node3["inputs"] and node3["inputs"][0]["count"] > 0:
        break
      time.sleep(0.05)
      next(ds_iter)

    node1 = stats
    node2 = node1["inputs"][0]
    node3 = node2["inputs"][0]
    node4 = node3["inputs"][0]
    node5 = node4["inputs"][0]
    node6 = node5["inputs"][0]
    self.assertIn("Interleave", node1["name"])
    self.assertLen(node1["inputs"], 3)
    self.assertIn("ThreadPrefetch", node2["name"])
    self.assertIn("ProcessPrefetch", node3["name"])
    self.assertIn("Prefetch", node4["name"])
    self.assertIn("Slice", node5["name"])
    self.assertIn("Range", node6["name"])

  def test_autotune_parameter_picklable(self):
    ds = grain.MapDataset.range(0, 1000)
    ds = ds.to_iter_dataset(
        read_options=options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=5, min_value=1, max_value=10
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=2, min_value=1, max_value=4
            ),
        )
    )
    ds = ds.mp_prefetch(options=options.MultiprocessingOptions(num_workers=2))
    ds = autotune.autotune(
        ds,
        model_config=bindings.AutotuneModelConfig(
            ram_budget_gb=4.0, cpu_budget=4
        ),
        allow_unknown_nodes=True,
    )
    ds_iter = ds.__iter__()
    next(ds_iter)
    next(ds_iter)

  def test_element_spec(self):
    ds = grain.MapDataset.range(10).to_iter_dataset()
    parent_spec = grain.experimental.get_element_spec(ds)
    autotune_ds = autotune.autotune(ds, allow_unknown_nodes=True)
    spec = grain.experimental.get_element_spec(autotune_ds)
    self.assertEqual(spec, parent_spec)

  def test_process_prefetch_tuning_in_worker(self):
    # Adding a Map that releases the GIL increases the likelihood
    # of the optimizer tuning number of threads or buffer size.
    def slow_map(x):
      time.sleep(0.001)
      return x

    ds = grain.MapDataset.range(10).repeat().map(slow_map)
    ds = ds.to_iter_dataset(
        read_options=options.ReadOptions(
            prefetch_buffer_size=bindings.AutotuneParameter(
                "buffer_size", initial_value=1, min_value=1, max_value=10
            ),
            num_threads=bindings.AutotuneParameter(
                "concurrency", initial_value=1, min_value=1, max_value=4
            ),
        )
    )
    ds = ds.mp_prefetch(options=options.MultiprocessingOptions(num_workers=1))

    model_config = bindings.AutotuneModelConfig(
        ram_budget_gb=4.0,
        cpu_budget=4,
        optimization_frequency=100,
        warmup_steps=0,
    )

    ds = autotune.autotune(
        ds,
        model_config=model_config,
        allow_unknown_nodes=True,
    )

    ds_iter = ds.__iter__()

    for _ in range(3000):
      next(ds_iter)
      interleave_node = ds_iter._autotune_node
      thread_prefetch_node = interleave_node.get_inputs()[0]
      process_prefetch_node = thread_prefetch_node.get_inputs()[0]
      if process_prefetch_node.get_inputs():
        prefetch_node = process_prefetch_node.get_inputs()[0]
        if prefetch_node.concurrency > 1 or prefetch_node.buffer_size > 1:
          # Parameters changed. Tuning was successful.
          return

    self.fail("Parameters did not change. Tuning was not successful.")

  def test_process_prefetch_large_buffer(self):
    """Test that mp_prefetch works with a large buffer size and many elements."""
    ds = grain.MapDataset.range(2000).to_iter_dataset()
    ds = ds.map(lambda x: b"a" * 100_000)
    ds = ds.mp_prefetch(
        options=options.MultiprocessingOptions(
            num_workers=1, per_worker_buffer_size=1000
        )
    )
    ds = autotune.autotune(
        ds,
        model_config=bindings.AutotuneModelConfig(
            ram_budget_gb=4.0, cpu_budget=4
        ),
        allow_unknown_nodes=True,
    )
    ds_iter = ds.__iter__()

    # Consume all elements.
    elements = list(ds_iter)
    self.assertLen(elements, 2000)
    self.assertEqual(elements[0], b"a" * 100_000)

    stats = ds_iter._autotune_node.get_pipeline_stats()
    self.assertGreater(stats["count"], 0)

  def test_interleave_cycle_length_tuned_correctly(self):
    def slow_map(x):
      time.sleep(0.01)
      return x

    ds = (
        grain.MapDataset.range(10)
        .map(slow_map)
        .to_iter_dataset(
            options.ReadOptions(num_threads=1, prefetch_buffer_size=2)
        )
    )

    cycle_length_param = bindings.AutotuneParameter(
        "cycle_length", initial_value=1.0, min_value=1.0, max_value=5.0
    )

    ds = interleave.InterleaveIterDataset(
        datasets=[ds] * 5, cycle_length=cycle_length_param
    )

    config = bindings.AutotuneModelConfig(
        ram_budget_gb=10.0,
        cpu_budget=10,
        optimization_frequency=1,
        warmup_steps=0,
    )
    autotune_ds = autotune.autotune(ds, model_config=config)

    it = iter(autotune_ds)

    self.assertEqual(cycle_length_param.get_value(), 1.0)

    # Iterate a few times to record stats and trigger optimization.
    for _ in range(5):
      next(it)

    # The model should want to increase cycle_length to reduce output time.
    self.assertGreater(cycle_length_param.get_value(), 1.0)

  def test_interleave_cycle_length_tuned_correctly_with_rebatch(self):
    def slow_map(x):
      time.sleep(0.01)
      return x

    ds = (
        grain.MapDataset.range(20)
        .map(slow_map)
        .to_iter_dataset(
            options.ReadOptions(num_threads=1, prefetch_buffer_size=2)
        )
        .batch(2)
    )

    cycle_length_param = bindings.AutotuneParameter(
        "cycle_length", initial_value=1.0, min_value=1.0, max_value=5.0
    )

    ds = interleave.InterleaveIterDataset(
        datasets=[ds] * 5, cycle_length=cycle_length_param
    )

    ds = grain.experimental.RebatchIterDataset(ds, batch_size=4)

    config = bindings.AutotuneModelConfig(
        ram_budget_gb=10.0,
        cpu_budget=10,
        optimization_frequency=1,
        warmup_steps=5,
    )
    autotune_ds = autotune.autotune(
        ds, model_config=config, allow_unknown_nodes=True
    )

    it = iter(autotune_ds)

    self.assertEqual(cycle_length_param.get_value(), 1.0)

    for _ in range(10):
      next(it)

    self.assertGreater(cycle_length_param.get_value(), 1.0)

  def test_autotune_concurrency_regularization_configured(self):
    ds = grain.MapDataset.range(10)
    iter_ds = ds.to_iter_dataset(
        options.ReadOptions(prefetch_buffer_size=1, num_threads=1)
    )
    autotune_ds = autotune.autotune(
        iter_ds,
        model_config=_get_test_model_config(),
        concurrency_reg_mode=bindings.ConcurrencyRegularizationMode.LINEAR,
        concurrency_reg_weight=0.02,
    )
    _ = iter(autotune_ds)
    self.assertIsNotNone(autotune_ds._autotune_node)
    self.assertEqual(
        autotune_ds._autotune_node.get_concurrency_regularization_mode(),
        bindings.ConcurrencyRegularizationMode.LINEAR,
    )
    self.assertEqual(
        autotune_ds._autotune_node.get_concurrency_regularization_weight(),
        0.02,
    )


if __name__ == "__main__":
  absltest.main()
