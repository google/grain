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
import json
from typing import Any, Optional, TypeVar

from etils import epath
from grain._src.core import sharding
from grain._src.python import data_loader
from grain._src.python.checkpoint import elastic_checkpoint
from grain._src.python.dataset import dataset
from grain._src.python.dataset import elastic_iterator

IteratorType = TypeVar(
    "IteratorType", data_loader.DataLoaderIterator, dataset.DatasetIterator
)


# Implements orbax.checkpoint.CheckpointHandler.
class CheckpointHandler:
  """Orbax CheckpointHandler for PyGrain iterators.

  This handler provides a bridge between Orbax checkpointing and PyGrain
  iterators. It manages the serialization and deserialization of iterator
  states across distributed processes, ensuring each process saves its unique
  shard state to a JSON file.

  The handler supports both the legacy `item` parameter (for backward
  compatibility with older Orbax API prior to 0.5.0, see
  https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html)
  and the newer Orbax V0 `args` API. It dynamically handles state formats:
  `dataset.DatasetIterator` types are serialized as JSON dictionaries, while
  other iterator types are handled as decoded strings.

  Examples:
    Refer to `CheckpointRestore` and `CheckpointSave` for examples.
  """

  def save(
      self,
      directory: epath.Path,
      # `item` is for backward compatibility with older Orbax API (prior to
      # 0.5.0), see
      # https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html.
      item: Optional[IteratorType] = None,
      # `args` is for the Orbax V0 API (since `orbax-checkpoint-0.5.0`). Note
      # that the Orbax V1 API does not use `args`.
      args: Any = None,
  ):
    """Saves the given iterator to the checkpoint in `directory`.

    Retrieves the internal state of the iterator, formats it appropriately
    (using JSON serialization for `DatasetIterator`s, and raw string decoding
    for other iterator types), and writes it to a process-specific file named
    `process_<index>-of-<count>.json` in the specified checkpoint directory.

    Args:
      directory: The directory where the checkpoint file will be written.
      item: The `DataLoaderIterator` or `DatasetIterator` to be saved. Should be
        provided if `args` is None. For backward compatibility with older Orbax
        API (prior to 0.5.0), see
        https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html.
      args: For the Orbax V0 API (since `orbax-checkpoint-0.5.0`), `args` will
        contain the item to save (typically as `args.item`). Should be provided
        if `item` is None. Note that the newest Orbax V1 API does not use `args`
        and instead operates on checkpointables directly.
    """
    item = item or args.item  # pytype:disable=attribute-error
    if isinstance(item, dataset.DatasetIterator):
      # ElasticIterDatasetIterator uses a custom checkpointing mechanism which
      # saves multiple files in the checkpoint directory. We should save the
      # state from all iterators in a single file.
      if isinstance(
          item,
          elastic_iterator.ElasticIterator,
      ) and isinstance(
          item.base_iterator, elastic_iterator.ElasticIterDatasetIterator
      ):
        elastic_checkpoint.save_elastic_iterator(directory, item)
      state = json.dumps(item.get_state(), indent=4)
    else:
      state = item.get_state().decode()
    process_index, process_count = sharding.get_process_index_and_count()
    filename = directory / f"process_{process_index}-of-{process_count}.json"
    filename.write_text(state)

  def restore(
      self,
      directory: epath.Path,
      # `item` is for backward compatibility with older Orbax API (prior to
      # 0.5.0), see
      # https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html.
      item: Optional[IteratorType] = None,
      # `args` is for the Orbax V0 API (since `orbax-checkpoint-0.5.0`). Note
      # that the Orbax V1 API does not use `args`.
      args: Any = None,
  ) -> IteratorType:
    """Restores the given iterator from the checkpoint in `directory`.

    Reads the process-specific JSON file from the checkpoint directory,
    deserializes the state, and applies it to the provided iterator using
    `set_state()`. The iterator will be modified in place. Critically, this
    method also triggers `start_prefetch()` after restoration to ensure the data
    pipeline resumes immediately.

    Args:
      directory: The directory containing the checkpoint file.
      item: A freshly created `DataLoaderIterator` or `DatasetIterator`. The
        state will be applied to this object. Should be provided if `args` is
        None. For backward compatibility with older Orbax API (prior to 0.5.0),
        see
        https://orbax.readthedocs.io/en/latest/guides/checkpoint/api_refactor.html.
      args: The Orbax V0 arguments container (typically a `CheckpointRestore`
        instance, since `orbax-checkpoint-0.5.0`) holding the target iterator.
        Should be provided if `item` is None. Note that the newest Orbax V1 API
        does not use `args` and instead operates on checkpointables directly.

    Returns:
      The restored `DataLoaderIterator` or `DatasetIterator` with restored state
      and active prefetching. This is the same object as the `item` argument
      (or `args.item`).

    Raises:
      ValueError: If the required process-specific checkpoint file does not
        exist.
    """
    item = item or args.item  # pytype:disable=attribute-error
    process_index, process_count = sharding.get_process_index_and_count()
    if isinstance(
        item,
        elastic_iterator.ElasticIterator,
    ) and isinstance(
        item.base_iterator, elastic_iterator.ElasticIterDatasetIterator
    ):
      # In the case of elastic iterators, we can restore from a checkpoint even
      # if the number of processes has changed. We check for this case and if
      # so we restore from shards.
      if process_count != elastic_checkpoint.get_checkpoint_process_count(
          directory
      ):
        elastic_checkpoint.restore_elastic_iterator(directory, item)
        return item  # pyrefly: ignore[bad-return]
    filename = directory / f"process_{process_index}-of-{process_count}.json"
    if not filename.exists():
      raise ValueError(f"File {filename} does not exist.")
    state = filename.read_text()
    if isinstance(item, dataset.DatasetIterator):
      state = json.loads(state)
    else:
      state = state.encode()
    item.set_state(state)  # pyrefly: ignore[bad-argument-type]
    item.start_prefetch()
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
  # Register the handler to be used with the Orbax v0 CheckpointManager API if
  # Orbax is present.
  import orbax.checkpoint as ocp  # v0 API; pylint:disable=g-import-not-at-top # pytype:disable=import-error

  @ocp.args.register_with_handler(CheckpointHandler, for_save=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class CheckpointSave(ocp.args.CheckpointArgs):
    """Arguments for saving a PyGrain iterator via Orbax.

    This dataclass registers the `CheckpointHandler` for save operations
    within the Orbax API. It wraps the iterator instance, allowing it to be
    passed to a `CheckpointManager` using the newer Orbax V0 `args` interface.

    Attributes:
      item: The iterator instance to be saved.

    Example:
      Saving the iterator alongside model weights using Orbax Composite::

        import orbax.checkpoint as ocp  # v0 API
        import grain

        from etils import epath
        import tempfile

        # Setup Data
        model_weights = {'layer1': 1.0, 'layer2': 2.0}
        ds = grain.MapDataset.range(100)
        iterator = iter(ds)
        for _ in range(10):
            next(iterator)

        # Setup Orbax CheckpointManager
        with tempfile.TemporaryDirectory() as temp_dir:
          path = epath.Path(temp_dir) / 'orbax_grain_checkpoint'
          path.mkdir(parents=True, exist_ok=True)

          with ocp.CheckpointManager(path) as mngr:
            mngr.save(
                step=0,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_weights),
                    data_iter=grain.checkpoint.CheckpointSave(item=iterator)
                )
            )
    """
    item: Any

  @ocp.args.register_with_handler(CheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
  @dataclasses.dataclass
  class CheckpointRestore(ocp.args.CheckpointArgs):
    """Arguments for restoring a PyGrain iterator via Orbax.

    This dataclass registers the `CheckpointHandler` for restore operations
    within the Orbax API. It wraps the iterator instance, allowing it to be
    passed to a `CheckpointManager` using the newer Orbax V0 `args` interface.

    Attributes:
      item: The freshly created iterator instance to restore into.

    Example:
      Restoring the iterator alongside model weights using Orbax Composite::

        import orbax.checkpoint as ocp  # v0 API
        import grain

        from etils import epath
        import tempfile

        # First, save a checkpoint to restore from.
        model_weights = {'layer1': 1.0, 'layer2': 2.0}
        ds = grain.MapDataset.range(100)
        save_iterator = iter(ds)
        for _ in range(10):
            next(save_iterator)

        with tempfile.TemporaryDirectory() as temp_dir:
          path = epath.Path(temp_dir) / 'orbax_grain_checkpoint'
          path.mkdir(parents=True, exist_ok=True)

          with ocp.CheckpointManager(path) as mngr:
            mngr.save(
                step=0,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(model_weights),
                    data_iter=grain.checkpoint.CheckpointSave(item=save_iterator)
                )
            )

          # Now, restore into fresh iterator and weights using a new manager
          # instance.
          restore_iterator = iter(ds)
          with ocp.CheckpointManager(path) as mngr:
            restored_data = mngr.restore(
                step=0,
                args=ocp.args.Composite(
                    model=ocp.args.StandardRestore(),
                    data_iter=grain.checkpoint.CheckpointRestore(item=restore_iterator)
                )
            )
          # Model weights are returned, iterator is restored in-place.
          print(restored_data['model'])
          # {'layer1': 1.0, 'layer2': 2.0}
          print(next(restore_iterator))
          # 10
    """
    item: Any

except (ImportError, TypeError, AttributeError):
  pass
