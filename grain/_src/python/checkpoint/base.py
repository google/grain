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
"""Utilities providing checkpointing capabilities for Grain iterators."""

import asyncio
from typing import Callable, Protocol
from etils import epath
from grain._src.core import sharding


class PathAwaitingCreation(Protocol):
  """A path that is in the process of being created.

  Please see orbax/checkpoint for the full definition of this type.
  """

  async def await_creation(self) -> epath.Path:
    """Waits for the path to be created.

    This function MUST be called before accessing the physical path. Prefer to
    perform in the background operation, rather than the main-thread-blocking
    operation.

    Returns:
      The path that is now created.
    """


async def background_save(directory: PathAwaitingCreation, state: str):
  """An async function that saves iterator state in a background thread.

  Args:
    directory: The directory to save the state to.
    state: The state to save.
  """
  directory = await directory.await_creation()
  process_index, process_count = sharding.get_process_index_and_count()
  filename = directory / f"process_{process_index}-of-{process_count}.json"
  await asyncio.to_thread(filename.write_text, state)


async def background_load(
    directory: epath.Path, set_state_fn: Callable[[str], None]
):
  process_index, process_count = sharding.get_process_index_and_count()
  filename = directory / f"process_{process_index}-of-{process_count}.json"
  if not await asyncio.to_thread(filename.exists):
    raise ValueError(f"File {filename} does not exist.")
  state = await asyncio.to_thread(filename.read_text)
  set_state_fn(state)
