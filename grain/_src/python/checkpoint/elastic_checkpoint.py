"""This module provides checkpointing logic for ElasticIterDatasetIterator."""

import dataclasses
import json
from typing import Any, Optional, Sequence

from etils import epath
from grain._src.python.dataset import elastic_iterator


def _find_shard_file(
    directory: epath.Path,
    shard_index: int,
    total_num_shards: int,
) -> epath.Path:
  """Finds all files matching 'shard_state_*.json' in the directory."""
  all_files = list(directory.iterdir())
  pattern = f"shard_state_{shard_index}-of-{total_num_shards}.json"
  found_files = [f for f in all_files if f.name.endswith(pattern)]
  if not found_files:
    raise ValueError(
        f"No shard state files found in {directory} for shard {shard_index}"
    )
  if len(found_files) > 1:
    raise ValueError(
        f"Multiple shard state files found in {directory} for shard"
        f" {shard_index}"
    )
  return found_files[0]


def save_elastic_iterator(
    directory: epath.Path,
    item: elastic_iterator.ElasticIterDatasetIterator,
):
  """Saves the given iterator to the checkpoint in `directory`."""
  state = item.get_state()
  ds_iterator_states = state["ds_iterator_states"]
  total_num_shards = state["total_num_shards"]
  for idx, host_iterator_state in ds_iterator_states.items():
    host_iterator_state["total_num_shards"] = total_num_shards
    shard_state = json.dumps(host_iterator_state, indent=4)
    filename = directory / f"shard_state_{idx}-of-{total_num_shards}.json"
    filename.write_text(shard_state)


def restore_elastic_iterator(
    directory: epath.Path,
    item: elastic_iterator.ElasticIterDatasetIterator,
):
  """Restores the given iterator from the checkpoint in `directory`."""
  total_num_shards = item.total_num_shards
  shard_index = item.shard_options.shard_index
  shard_count = item.shard_options.shard_count
  while shard_index < total_num_shards:
    filename = _find_shard_file(directory, shard_index, total_num_shards)
    state = filename.read_text()
    state = json.loads(state)
    item.update_shard_iterator_state(shard_index, state)
    shard_index += shard_count


class ElasticCheckpointHandler:
  """Orbax CheckpointHandler for PyGrain iterators."""

  def save(
      self,
      directory: epath.Path,
      item: Optional[
          elastic_iterator.ElasticIterDatasetIterator
          | Sequence[elastic_iterator.ElasticIterDatasetIterator]
      ] = None,
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`."""
    item = item or args.item
    if isinstance(item, elastic_iterator.ElasticIterDatasetIterator):
      item = [item]
    for iterator in item:
      save_elastic_iterator(directory, iterator)

  def restore(
      self,
      directory: epath.Path,
      item: Optional[
          elastic_iterator.ElasticIterDatasetIterator
          | Sequence[elastic_iterator.ElasticIterDatasetIterator]
      ] = None,
      args: Any = None,
  ) -> Any:
    """Restores the given iterator from the checkpoint in `directory`."""
    item = item or args.item
    if isinstance(item, elastic_iterator.ElasticIterDatasetIterator):
      item = [item]
    for iterator in item:
      restore_elastic_iterator(directory, iterator)
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

  @classmethod
  def typestr(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


try:
  # Register the handler to be used with the new checkpointing API if Orbax is
  # present.
  import orbax.checkpoint as ocp  # pylint:disable=g-import-not-at-top # pytype:disable=import-error

  @ocp.args.register_with_handler(ElasticCheckpointHandler, for_save=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class ElasticCheckpointSave(ocp.args.CheckpointArgs):
    item: Any

  @ocp.args.register_with_handler(ElasticCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class ElasticCheckpointRestore(ocp.args.CheckpointArgs):
    item: Any

except (ImportError, TypeError, AttributeError):
  pass
