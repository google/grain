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
"""This module provides a PyGrain CheckpointHandler for integration with Orbax."""
from typing import Any, Optional

from etils import epath
from grain._src.python import data_loader
import jax

PyGrainDatasetIterator = data_loader.PyGrainDatasetIterator


# Ipmlements orbax.checkpoint.CheckpointHandler.
class PyGrainCheckpointHandler:
  """Orbax CheckpointHandler for PyGrainDatasetIterator."""

  def save(self, directory: epath.Path, item: PyGrainDatasetIterator):
    filename = (
        directory
        / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    )
    filename.write_text(item.get_state().decode())

  def restore(
      self, directory: epath.Path, item: Optional[PyGrainDatasetIterator] = None
  ) -> PyGrainDatasetIterator:
    """Restores the given iterator from the checkpoint in `directory`."""
    if item is None:
      raise ValueError("OrbaxCheckpointHandler requires an `item`.")
    filename = (
        directory
        / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    )
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    state = filename.read_text().encode()
    item.set_state(state)
    return item

  # Required by interface but not supported by PyGrain checkpoints.
  def structure(self, directory: epath.Path) -> Any:
    del directory
    return None
