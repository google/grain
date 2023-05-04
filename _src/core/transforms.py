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
"""Abstract base classes for common types of transformations.

The idea is to implement atomic chunks of transformations as subclasses of
the base classes below (examples: resize image, tokenize text, add padding).
- More complex transformations can be created by chaining multiple
  transformations.
- It's recommended that subclasses are dataclasses.
- Libraries applying transformations can use these base classes to correctly
  apply transformations as part of Beam pipelines, data ingestion pipelines etc.
"""

import abc
from typing import Sequence, Union

import numpy as np


class MapTransform(abc.ABC):
  """Abstract base class for all 1:1 transformations of elements."""

  @abc.abstractmethod
  def map(self, element):
    """Maps a single element."""


class RandomMapTransform(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements."""

  @abc.abstractmethod
  def random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""


class TfRandomMapTransform(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements."""

  @abc.abstractmethod
  def np_random_map(self, element, rng: np.random.Generator):
    """Maps a single element."""


class FilterTransform(abc.ABC):
  """Abstract base class for filter transformations for individual elements."""

  @abc.abstractmethod
  def filter(self, element) -> bool:
    """Filters a single element."""


Transformation = Union[
    MapTransform, RandomMapTransform, TfRandomMapTransform, FilterTransform
]
Transformations = Sequence[Transformation]
