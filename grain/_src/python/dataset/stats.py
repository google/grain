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
"""Tools for recording statistics about dataset transformations."""

from __future__ import annotations

import abc
import contextlib
from typing import Sequence, TypeVar


T = TypeVar("T")


class Stats(abc.ABC):
  """Base abstract class for statistics recording.

  This class replicates the transformation tree structure and provides
  interfaces for recording statistics in the given transformation node.
  """

  def __init__(self, name: str, parents: Sequence[Stats]):
    self._name = name
    self._parents = parents
    # Mark parent nodes as non-outputs. Nodes that are not updated are the
    # output nodes.
    self._is_output = True
    for p in parents:
      p._is_output = False

  @contextlib.contextmanager
  @abc.abstractmethod
  def record_self_time(self):
    """Records time spent in this node's transfromation.

    Implemented as context manager for convenience. Expected to be used as
    follows:
    ```
    class MyMapDataset(MapDataset):
      ...
      def __getitem__(self, index):
        input_element = self._parent[index]
        with self._stats.record_self_time():
          return self._map_fn(input_element)
    ```
    and
    ```
    class MyMapDatasetIterator(DatasetIterator):
      ...
      def __next__(self):
        input_element = next(self._parent)
        with self._stats.record_self_time():
          return self._map_fn(input_element)
    ```
    """
    ...

  @abc.abstractmethod
  def record_output_spec(self, element: T) -> T:
    """Records output spec of the elements produced by this node.

    Args:
      element: structure to record the spec of.

    Returns: the `element` unchanged (for convenience).

    Expected to be used as follows:
    ```
    class MyMapDataset(MapDataset):
      ...
      def __getitem__(self, index):
        input_element = self._parent[index]
        return self._stats.record_output_spec(self._map_fn(input_element))
    ```
    and
    ```
    class MyMapDatasetIterator(DatasetIterator):
      ...
      def __next__(self):
        input_element = next(self._parent)
        return self._stats.record_output_spec(self._map_fn(input_element))
    ```
    """
    ...

  @abc.abstractmethod
  def report(self):
    """Reports the collected statistics.

    This should be expected to be called once the last element is processed as
    well as in the middle of execution.
    """
    ...


class NoopStats(Stats):
  """Default implementation for statistics collection that does nothing."""

  @contextlib.contextmanager
  def record_self_time(self):
    yield

  def record_output_spec(self, element: T) -> T:
    return element

  def report(self):
    pass
