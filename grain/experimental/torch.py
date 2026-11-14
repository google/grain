# Copyright 2026 Google LLC
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
"""Experimental PyTorch-first Grain APIs (`grain.experimental.torch`).

PyTorch remains an optional dependency: importing `grain` or
`grain.experimental.torch` never eagerly imports `torch`.
"""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-multiple-import
# pylint: disable=unused-import

from grain._src.python.dataset.base import (
    DatasetOptions,
)
from grain._src.python.experimental.torch.checkpoint import (
    IteratorState,
)
from grain._src.python.experimental.torch.distributed import (
    get_process_index_and_count,
)
from grain._src.python.experimental.torch.options import (
    MultiprocessingOptions,
)
from grain._src.python.experimental.torch.prefetch import (
    mp_prefetch,
)
from grain._src.python.experimental.torch.random_map import (
    TorchStatelessRandomMap,
)
