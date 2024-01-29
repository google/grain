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
import dataclasses
from typing import Any, Optional

from etils import epath
from grain._src.python import data_loader
import jax

PyGrainDatasetIterator = data_loader.PyGrainDatasetIterator


# Ipmlements orbax.checkpoint.CheckpointHandler.
class PyGrainCheckpointHandler:
  """Orbax CheckpointHandler for PyGrainDatasetIterator."""

  def save(
      self,
      directory: epath.Path,
      # `item` is for backwards compatibility with older Orbax API, see
      # https://orbax.readthedocs.io/en/latest/api_refactor.html.
      item: Optional[PyGrainDatasetIterator] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error
    filename = (
        directory
        / f"process_{jax.process_index()}-of-{jax.process_count()}.json"
    )
    filename.write_text(item.get_state().decode())

  def restore(
      self,
      directory: epath.Path,
      item: Optional[PyGrainDatasetIterator] = None,
      args: Any = None,
  ) -> PyGrainDatasetIterator:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item  # pytype:disable=attribute-error
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

  # Required by interface.

  def metadata(self, directory: epath.Path) -> Optional[Any]:
    del directory
    return None

  def finalize(self, directory: epath.Path):
    pass

  def close(self):
    pass


try:
  # Register the handler to be used with the new checkpointing API if Orbax is
  # present.
  import orbax.checkpoint as ocp  # pylint:disable=g-import-not-at-top # pytype:disable=import-error

  @ocp.args.register_with_handler(PyGrainCheckpointHandler, for_save=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class PyGrainCheckpointSave(ocp.args.CheckpointArgs):
    item: Any

  @ocp.args.register_with_handler(PyGrainCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class PyGrainCheckpointRestore(ocp.args.CheckpointArgs):
    item: Any

except ImportError:
  pass
