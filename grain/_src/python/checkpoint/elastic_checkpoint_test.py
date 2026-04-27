"""Tests for elastic checkpoint."""

import json

from etils import epath
from grain._src.core import sharding
from grain._src.python.checkpoint import elastic_checkpoint
from grain._src.python.dataset import elastic_iterator

from absl.testing import absltest


class MockElasticIterDatasetIterator(
    elastic_iterator.ElasticIterDatasetIterator
):

  def __init__(self, shard_options, total_num_shards, states=None):
    self._shard_options = shard_options
    self._num_dataset_shards = total_num_shards
    self._states = states if states is not None else {}
    self.updated_states = {}

  def get_state(self):
    return {
        "ds_iterator_states": self._states,
    }

  def set_state(self, state):
    for k, v in state["ds_iterator_states"].items():
      self.updated_states[k] = v


class ElasticCheckpointTest(absltest.TestCase):

  def test_save_and_restore_elastic_iterator(self):
    temp_dir = epath.Path(self.create_tempdir().full_path)
    shard_options = sharding.ShardOptions(shard_index=0, shard_count=1)
    states = {
        0: {"val": 0},
        1: {"val": 1},
    }
    iterator = MockElasticIterDatasetIterator(
        shard_options=shard_options, total_num_shards=2, states=states
    )
    elastic_checkpoint.save_elastic_iterator(temp_dir, iterator)

    file0 = temp_dir / "shard_state_0.json"
    self.assertTrue(file0.exists())
    self.assertEqual(
        file0.read_text(),
        json.dumps({"val": 0}, indent=4),
    )
    file1 = temp_dir / "shard_state_1.json"
    self.assertTrue(file1.exists())
    self.assertEqual(
        file1.read_text(),
        json.dumps({"val": 1}, indent=4),
    )

    iterator_to_restore = MockElasticIterDatasetIterator(
        shard_options=shard_options, total_num_shards=2
    )
    elastic_checkpoint.restore_elastic_iterator(temp_dir, iterator_to_restore)
    self.assertEqual(
        iterator_to_restore.updated_states,
        {
            0: {"val": 0},
            1: {"val": 1},
        },
    )

  def test_restore_elastic_iterator_with_multiple_processes(self):
    temp_dir = epath.Path(self.create_tempdir().full_path)
    # Process 0
    shard_options_0 = sharding.ShardOptions(shard_index=0, shard_count=2)
    states = {
        0: {"val": 0},
        1: {"val": 1},
        2: {"val": 2},
    }
    iterator_0 = MockElasticIterDatasetIterator(
        shard_options=shard_options_0, total_num_shards=3, states=states
    )
    # In reality save_elastic_iterator will be called in each process, but
    # get_state() should return all states, so we only need to call it once
    # to create checkpoint files.
    elastic_checkpoint.save_elastic_iterator(temp_dir, iterator_0)

    # Check files are written
    self.assertTrue((temp_dir / "shard_state_0.json").exists())
    self.assertTrue((temp_dir / "shard_state_1.json").exists())
    self.assertTrue((temp_dir / "shard_state_2.json").exists())

    # Restore for process 0, responsible for shards 0 and 2.
    iterator_to_restore_0 = MockElasticIterDatasetIterator(
        shard_options=shard_options_0, total_num_shards=3
    )
    elastic_checkpoint.restore_elastic_iterator(temp_dir, iterator_to_restore_0)
    self.assertEqual(
        iterator_to_restore_0.updated_states,
        {
            0: {"val": 0},
            2: {"val": 2},
        },
    )

    # Restore for process 1, responsible for shard 1.
    shard_options_1 = sharding.ShardOptions(shard_index=1, shard_count=2)
    iterator_to_restore_1 = MockElasticIterDatasetIterator(
        shard_options=shard_options_1, total_num_shards=3
    )
    elastic_checkpoint.restore_elastic_iterator(temp_dir, iterator_to_restore_1)
    self.assertEqual(
        iterator_to_restore_1.updated_states,
        {
            1: {"val": 1},
        },
    )


if __name__ == "__main__":
  absltest.main()
