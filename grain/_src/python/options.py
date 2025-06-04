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
"""Dataclasses for holdings options."""
import dataclasses


@dataclasses.dataclass(slots=True)
class ReadOptions:
  """Options for reading data from the DataSource.

  These settings configure a single Python process. Each process uses separate
  threads and buffer for reading and processing data.

  Example: With ReadOptions.num_threads=8 and
  MultiprocessingOptions.num_workers=10 there will be 80 threads reading the
  data (8 threads in each of 10 Python processes).

  Attributes:
    num_threads: Number of threads reading from the DataSource in parallel. If
      the data are already loaded in memory, we recommend setting this to 0 to
      avoid Python GIL contention by multiple threads.
    prefetch_buffer_size: Size of the buffer for reading elements per Python
      process (not per thread). Useful when reading from a distributed file
      system.
  """

  # The current default values where chosen by running a few selected
  # benchmarks reading from remote hard drives.
  # These values should work well for datasets with elements between 1 and
  # 10 KiB on disk.
  num_threads: int = 16
  prefetch_buffer_size: int = 500


@dataclasses.dataclass(slots=True)
class MultiprocessingOptions:
  """Options for using Python multiprocessing.

  Attributes:
    num_workers: Number of Python worker processes. More processes can speed up
      the pipeline if it's compute bound and bottlenecked on the CPython's GIL.
      The default value of 0 means no Python multiprocessing, and as a result
      all data loading and transformation will run in the main Python process.
    per_worker_buffer_size: Size of the buffer for preprocessed elements that
      each worker maintains. These are elements after all transformations. If
      your transformations include batching this means a single element is a
      batch.
    enable_profiling: If True, profiling info is logged. This is only available
      when num_workers >= 1.
  """

  num_workers: int = 0
  per_worker_buffer_size: int = 1
  enable_profiling: bool = False
