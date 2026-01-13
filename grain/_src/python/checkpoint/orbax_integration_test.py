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
"""Tests for checkpoint handlers."""
from grain._src.core import sharding
from grain._src.python import data_loader
from grain._src.python import data_sources
from grain._src.python import samplers
from grain._src.python.dataset import dataset
from orbax.checkpoint import v1 as ocp

from absl.testing import absltest


class CheckpointHandlersTest(absltest.TestCase):

  def _create_data_loader(self) -> data_loader.DataLoader:
    # Generates elements [0, 1, 2, 3, 4, 5, 6, 7].
    range_data_source = data_sources.RangeDataSource(0, 8, 1)
    sampler = samplers.SequentialSampler(
        num_records=len(range_data_source),
        shard_options=sharding.NoSharding(),
    )
    return data_loader.DataLoader(
        data_source=range_data_source,
        sampler=sampler,
    )

  def _create_data_loader_iter_to_checkpoint(self):
    ds = self._create_data_loader()
    break_at = 4
    ds_iter = iter(ds)
    for _ in range(break_at):
      _ = next(ds_iter)
    return ds_iter

  def _assert_restored_data_loader_iter(self, ds_iter):
    expected_data = list(self._create_data_loader_iter_to_checkpoint())
    self.assertEqual(list(ds_iter), expected_data)

  def _create_dataset(self):
    return (
        dataset.MapDataset.range(35)
        .seed(23)
        .map(lambda x: x + 100)
        .shuffle()
        .to_iter_dataset()
        .batch(3)
        .map(lambda x: x.tolist())
    )

  def _create_dataset_iter_to_checkpoint(self):
    ds = self._create_dataset()
    break_at = 5
    ds_iter = iter(ds)
    for _ in range(break_at):
      _ = next(ds_iter)
    return ds_iter

  def _assert_restored_dataset_iter(self, ds_iter):
    expected_data = list(self._create_dataset_iter_to_checkpoint())
    self.assertEqual(list(ds_iter), expected_data)

  def test_data_loader_checkpoint_save_and_restore(self):
    tmpdir = f"{self.create_tempdir().full_path}/checkpoint"
    ocp.save_checkpointables(
        tmpdir, dict(dataset=self._create_data_loader_iter_to_checkpoint())
    )
    ds = self._create_data_loader()
    ds_iter = iter(ds)
    ocp.load_checkpointables(tmpdir, dict(dataset=ds_iter))
    self._assert_restored_data_loader_iter(ds_iter)

  def test_dataset_checkpoint_save_and_restore(self):
    tmpdir = f"{self.create_tempdir().full_path}/checkpoint"
    ocp.save_checkpointables(
        tmpdir, dict(dataset=self._create_dataset_iter_to_checkpoint())
    )
    ds = self._create_dataset()
    ds_iter = iter(ds)
    ocp.load_checkpointables(tmpdir, dict(dataset=ds_iter))
    self._assert_restored_dataset_iter(ds_iter)

  def test_composite_checkpoint_save_and_restore(self):
    tmpdir = f"{self.create_tempdir().full_path}/checkpoint"
    ocp.save_checkpointables(
        tmpdir,
        dict(
            state={"values": [0]},
            dataset=self._create_data_loader_iter_to_checkpoint(),
        ),
    )
    ds = self._create_data_loader()
    ds_iter = iter(ds)
    ocp.load_checkpointables(tmpdir, dict(state=None, dataset=ds_iter))
    self._assert_restored_data_loader_iter(ds_iter)


if __name__ == "__main__":
  absltest.main()
