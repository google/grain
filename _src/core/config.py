# Copyright 2022 Google LLC
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
"""Handle Grain config options.

Config options can be set via flags starting with '--grain_' or by calling
`grain.config.update(name, value)`.
"""
from typing import Any

from absl import flags

# Performance optimisations. Consider most of these experimental. We might
# remove them once we are confident that the default values work well for
# everyone.
_INTERLEAVED_SHUFFLE = flags.DEFINE_bool(
    "grain_tf_interleaved_shuffle", False,
    "If enable replace the global shuffle in index dataset with an "
    "approximation that interleaves several small blocks. This can increase "
    "throughput of the input pipeline.")
_LOOKUP_BATCH_SIZE = flags.DEFINE_integer(
    "grain_tf_lookup_batch_size", 1024,
    "Number of keys to batch into a single lookup call. "
    "--grain_tf_num_parallel_calls controls on how many lookups to perform in "
    "parallel.")
_LOOKUP_NUM_PARALLEL_CALLS = flags.DEFINE_integer(
    "grain_tf_lookup_num_parallel_calls", -1,
    "Number of parallel lookup calls. Each lookup will read "
    "--grain_tf_lookup_batch_size records. This must be a positive integer > 0 "
    "or -1. If -1 Grain will use tf.data.AUTOTUNE to tune the value "
    "automatically.")

_GRAIN_FLAGS = (_INTERLEAVED_SHUFFLE, _LOOKUP_BATCH_SIZE,
                _LOOKUP_NUM_PARALLEL_CALLS)


class Config:
  """Class for holding current Grain configuration."""

  # Loosen the static type checking requirements.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __getattr__(self, name: str) -> Any:
    flag_name = f"grain_{name}"
    if any(f.name == flag_name for f in _GRAIN_FLAGS):
      return getattr(flags.FLAGS, flag_name)
    raise ValueError(f"Unrecognized config option: {name}")

  def __setattr__(self, name: str, value: Any):
    raise ValueError("Please use update().")

  def update(self, name: str, value: Any):
    flag_name = f"grain_{name}"
    if any(f.name == flag_name for f in _GRAIN_FLAGS):
      setattr(flags.FLAGS, flag_name, value)
      return
    raise ValueError(f"Unrecognized config option: {name}")


config = Config()
