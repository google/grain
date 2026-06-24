"""This module provides checkpointing logic for ElasticIterDatasetIterator."""

import json

from etils import epath
from grain._src.python.dataset import elastic_iterator


def _find_shard_file(
    directory: epath.Path,
    shard_index: int,
) -> epath.Path | None:
  """Finds all files matching 'shard_state_*.json' in the directory."""
  file_path = directory / f"shard_state_{shard_index}.json"
  if file_path.exists():
    return file_path
  return None


def get_checkpoint_process_count(directory: epath.Path) -> int:
  """Finds the number of processes used to save the checkpoint."""
  process_count = 0
  for file_path in directory.glob("process_*-of-*.json"):
    process_count = max(process_count, int(file_path.stem.split("-of-")[1]))
  return process_count


def save_elastic_iterator(
    directory: epath.Path,
    item: elastic_iterator.ElasticIterDatasetIterator,
) -> None:
  """Saves the given iterator to the checkpoint in `directory`."""
  state = item.get_shard_states()
  for idx, host_iterator_state in state.items():
    shard_state = json.dumps(host_iterator_state, indent=4)
    filename = directory / f"shard_state_{idx}.json"
    filename.write_text(shard_state)


def restore_elastic_iterator(
    directory: epath.Path,
    item: elastic_iterator.ElasticIterDatasetIterator,
) -> None:
  """Restores the given iterator from the checkpoint in `directory`."""
  shard_index = item.shard_options.shard_index
  shard_count = item.shard_options.shard_count
  iterator_states = {}
  # We don't necessarily know how many shards per file we have since the number
  # of shards can be split unevenly between hosts. So we continue to add states
  # until we can't find any more files.
  while True:
    filename = _find_shard_file(directory, shard_index)
    if filename is None:
      break
    state = filename.read_text()
    state = json.loads(state)
    iterator_states[shard_index] = state
    shard_index += shard_count
  item.set_shard_states(iterator_states)
