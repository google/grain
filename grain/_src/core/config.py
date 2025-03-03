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
"""Handle Grain config options.

Config options can be set via flags starting with '--grain_' or by calling
`grain.config.update(name, value)`.
"""
from typing import Any

from absl import flags
from grain._src.core import monitoring as grain_monitoring

from grain._src.core import monitoring

# Performance optimisations. Consider most of these experimental. We might
# remove them once we are confident that the default values work well for
# everyone.
_INTERLEAVED_SHUFFLE = flags.DEFINE_bool(
    "grain_tf_interleaved_shuffle",
    False,
    (
        "If enabled replace the global shuffle with an approximation that "
        "interleaves small blocks. This can increase throughput of the input "
        "pipeline. See index_dataset._interleaved_shuffle() for details."
    ),
)
_INTERLEAVED_SHUFFLE_BLOCK_SIZE = flags.DEFINE_integer(
    "grain_tf_interleaved_shuffle_block_size",
    10,
    "Block size in index_dataset._interleaved_shuffle().",
)
_INTERLEAVED_SHUFFLE_NUM_BLOCKS_PER_PARTITION = flags.DEFINE_integer(
    "grain_tf_interleaved_shuffle_parallel_blocks",
    10,
    "Number of blocks per partition in index_dataset._interleaved_shuffle().",
)

_LOOKUP_BATCH_SIZE = flags.DEFINE_integer(
    "grain_tf_lookup_batch_size",
    100,
    (
        "Number of keys to batch into a single lookup call."
        " --grain_tf_num_parallel_calls controls on how many lookups to perform"
        " in parallel."
    ),
)
_LOOKUP_NUM_PARALLEL_CALLS = flags.DEFINE_integer(
    "grain_tf_lookup_num_parallel_calls",
    -1,
    (
        "Number of parallel lookup calls. Each lookup will read"
        " --grain_tf_lookup_batch_size records. This must be a positive integer"
        " > 0 or -1. If -1 Grain will use tf.data.AUTOTUNE to tune the value"
        " automatically."
    ),
)
_LOOKUP_FAST_WARMUP = flags.DEFINE_bool(
    "grain_tf_lookup_fast_warmup",
    False,
    (
        "Deprecated. If True will split up the first batch lookup into smaller"
        " chunks. This can help to provide initial elements fast while allowing"
        " when using a large tf_lookup_batch_size (which is usually better for"
        " throughput)."
    ),
)

_OPTIMIZED_SEQUENTIAL_READ = flags.DEFINE_bool(
    "grain_tf_optimized_sequential_read",
    False,
    (
        "Enables optimized reads in TfMixtureDataLoader in case when sampler"
        " produces sequential indices and the input sources support it."
        " Experimental -- do not use."
    ),
)

_PREFETCH_BUFFER_SIZE = flags.DEFINE_integer(
    "grain_tf_prefetch_buffer_size",
    -1,  # tf.data.AUTOTUNE
    (
        "Sets the prefetch buffer size for TfDataLoader/TfMixtureDataLoader"
        " dataset. This allows later elements to be prepared while the current"
        " element is being processed. This often improves latency and"
        " throughput, at the cost of using additional memory to store"
        " prefetched elements. This number represents the maximum number of"
        " elements that will be buffered when prefetching. Defaults -1 is"
        " equivalent to tf.data.AUTOTUNE where the buffer size is dynamically"
        " tuned."
    ),
)

_DEBUG_MODE = flags.DEFINE_bool(
    "grain_py_debug_mode",
    False,
    (
        "If True, will enable debug mode for Grain. This will enable reporting"
        "extra streamz metrics."
    ),
)

_DATASET_VISUALIZATION_OUTPUT_DIR = flags.DEFINE_string(
    "grain_py_dataset_visualization_output_dir",
    None,
    "If set, generates the pipeline visulization graph in the logs.",
)

_RELAY_SIGTERM_TO_MAIN = flags.DEFINE_bool(
    "grain_py_relay_sigterm_to_main",
    False,
    "If True, grain workers will relay SIGTERM to the main process.",
)

_GRAIN_FLAGS = (
    _INTERLEAVED_SHUFFLE,
    _INTERLEAVED_SHUFFLE_BLOCK_SIZE,
    _INTERLEAVED_SHUFFLE_NUM_BLOCKS_PER_PARTITION,
    _LOOKUP_BATCH_SIZE,
    _LOOKUP_NUM_PARALLEL_CALLS,
    _LOOKUP_FAST_WARMUP,
    _OPTIMIZED_SEQUENTIAL_READ,
    _PREFETCH_BUFFER_SIZE,
    _DEBUG_MODE,
    _DATASET_VISUALIZATION_OUTPUT_DIR,
    _RELAY_SIGTERM_TO_MAIN,
)

_grain_experiment_metric = monitoring.Metric(
    "/grain/experiment",
    value_type=int,
    metadata=monitoring.Metadata(description="Grain experiment opt-in metric."),
    root=grain_monitoring.get_monitoring_root(),
    fields=[("name", str)],
)


class Config:
  """Class for holding current Grain configuration."""

  # Loosen the static type checking requirements.
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __getattr__(self, name: str) -> Any:
    flag_name = f"grain_{name}"
    if any(f.name == flag_name for f in _GRAIN_FLAGS):
      value = getattr(flags.FLAGS, flag_name)
      if isinstance(value, int):
        int_value = value
      else:
        int_value = int(value != flags.FLAGS[flag_name].default)
      _grain_experiment_metric.Set(int_value, flag_name)
      return value
    raise ValueError(f"Unrecognized config option: {name}")

  def get_or_default(self, name: str) -> Any:
    """Returns the value if flags are parsed or the default value."""
    try:
      return self.__getattr__(name)
    except flags.UnparsedFlagAccessError:
      flag_name = f"grain_{name}"
      return flags.FLAGS[flag_name].default

  def __setattr__(self, name: str, value: Any):
    raise ValueError("Please use update().")

  def update(self, name: str, value: Any):
    flag_name = f"grain_{name}"
    if any(f.name == flag_name for f in _GRAIN_FLAGS):
      setattr(flags.FLAGS, flag_name, value)
      return
    raise ValueError(f"Unrecognized config option: {name}")


config = Config()
