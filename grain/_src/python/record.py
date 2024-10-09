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
"""Define record class used by various modules in the Grain Python Backend."""

import dataclasses
from typing import Generic, Optional, TypeVar
import numpy as np

T = TypeVar("T")


@dataclasses.dataclass(slots=True)
class RecordMetadata:
  """RecordMetadata contains metadata about indidivual records.

  Metadata can be emitted by the sampler to refer to which record to read next.
  In addition, they are also used to keep information about records as they flow
  through the pipeline from one operation to the other.
  """

  index: int
  record_key: Optional[int] = None
  rng: Optional[np.random.Generator] = None

  def remove_record_key(self):
    """Removes record key if exists."""
    if self.record_key is None:
      return self
    else:
      return dataclasses.replace(self, record_key=None)

  def __str__(self):
    return (
        f"RecordMetadata(index={self.index}, record_key={self.record_key}, "
        f"rng={self.rng})"
    )


@dataclasses.dataclass(slots=True)
class Record(Generic[T]):
  metadata: RecordMetadata
  data: T
