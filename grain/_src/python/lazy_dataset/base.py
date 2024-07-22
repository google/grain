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
"""Base interfaces for working with LazyDataset.

Classes in this module are shared by LazyDataset classes and transformations.
"""

import abc
import typing
from typing import Protocol, TypeVar


T = TypeVar("T")


@typing.runtime_checkable
class RandomAccessDataSource(Protocol[T]):
  """Interface for datasets where storage supports efficient random access."""

  def __len__(self):
    ...

  def __getitem__(self, index: int) -> T:
    ...


class DatasetSelectionMap(abc.ABC):
  """Map from index to (constituent dataset index, index within dataset).

  Note, this must be stateless, picklable and should avoid randomness to
  support determinism since it may be created and called concurrently in
  multiple processes.
  """

  @abc.abstractmethod
  def __len__(self) -> int:
    """Returns the length of this dataset."""

  @abc.abstractmethod
  def __getitem__(self, index: int) -> tuple[int, int]:
    """Returns constituent dataset index and index within this dataset."""
