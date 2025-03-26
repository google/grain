# Copyright 2025 Google LLC
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
"""APIs for saving and restoring pipeline state."""


# pylint: disable=g-importing-member
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
# pylint: disable=g-multiple-import

from grain._src.python.checkpoint_handlers import (
    CheckpointHandler,
)

# These are imported only if Orbax is present.
try:
  from grain._src.python.checkpoint_handlers import (
      CheckpointSave,
      CheckpointRestore,
  )
except ImportError:
  pass
