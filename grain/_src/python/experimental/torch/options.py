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
"""Options dataclasses for PyTorch pipelines in Grain."""

from __future__ import annotations

import dataclasses

from grain._src.python import options as grain_options


@dataclasses.dataclass(slots=True)
class MultiprocessingOptions(grain_options.MultiprocessingOptions):
  """Options for Python multiprocessing in PyTorch pipelines.

  Extends ``grain.MultiprocessingOptions`` with ``worker_intra_op_threads``
  (default ``1``) so PyTorch workers do not oversubscribe host CPU cores by
  default, like ``torch.utils.data.DataLoader`` workers.

  Attributes:
    worker_intra_op_threads: Intra-op thread count of each worker, applied with
      ``torch.set_num_threads`` (and ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``)
      before ``worker_init_fn`` runs. Set this field to use a different count:
      some PyTorch builds only honor the first ``torch.set_num_threads`` call,
      so a later call in ``worker_init_fn`` may be ignored. Pass ``None`` to
      leave worker threads unchanged.
  """

  worker_intra_op_threads: int | None = 1

  def __post_init__(self):
    if (
        self.worker_intra_op_threads is not None
        and self.worker_intra_op_threads <= 0
    ):
      raise ValueError(
          "`worker_intra_op_threads` must be a positive integer or None, got"
          f" {self.worker_intra_op_threads}."
      )
