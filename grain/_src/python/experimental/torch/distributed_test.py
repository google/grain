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

import os
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grain
from grain._src.python.experimental.torch import distributed as torch_distributed

dist: Any = None
try:
  import torch.distributed as dist  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

_get_process_index_and_count = torch_distributed.get_process_index_and_count


class ProcessIndexAndCountTest(parameterized.TestCase):

  def test_exported_in_grain_experimental_torch(self):
    self.assertIs(
        grain.experimental.torch.get_process_index_and_count,
        _get_process_index_and_count,
    )

  def test_single_process_defaults_to_zero_of_one(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("RANK", None)
      os.environ.pop("WORLD_SIZE", None)
      self.assertEqual(_get_process_index_and_count(), (0, 1))

  @parameterized.named_parameters(
      dict(
          testcase_name="valid_rank_2_of_4",
          env={"RANK": "2", "WORLD_SIZE": "4"},
          expected=(2, 4),
      ),
      dict(
          testcase_name="malformed_non_int",
          env={"RANK": "abc", "WORLD_SIZE": "4"},
          expected=(0, 1),
      ),
      dict(
          testcase_name="out_of_bounds_rank",
          env={"RANK": "4", "WORLD_SIZE": "4"},
          expected=(0, 1),
      ),
      dict(
          testcase_name="zero_world",
          env={"RANK": "0", "WORLD_SIZE": "0"},
          expected=(0, 1),
      ),
  )
  def test_env_fallback_when_uninitialized(
      self, env: dict[str, str], expected: tuple[int, int]
  ):
    with mock.patch.dict(os.environ, env, clear=False):
      self.assertEqual(_get_process_index_and_count(), expected)

  def test_explicit_group_without_initialized_pg_raises(self):
    with self.assertRaisesRegex(RuntimeError, "must be initialized"):
      _get_process_index_and_count(group=object())

  @absltest.skipIf(dist is None, "torch.distributed is not available.")
  def test_initialized_gloo_group_takes_precedence_over_env(self):
    store = dist.HashStore()
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
      with mock.patch.dict(
          os.environ, {"RANK": "3", "WORLD_SIZE": "8"}, clear=False
      ):
        self.assertEqual(_get_process_index_and_count(), (0, 1))
    finally:
      dist.destroy_process_group()

  def test_non_member_group_rank_negative_one_raises(self):
    fake_dist = mock.MagicMock()
    fake_dist.is_available.return_value = True
    fake_dist.is_initialized.return_value = True
    fake_dist.get_rank.return_value = -1
    fake_dist.get_world_size.return_value = 2
    fake_torch = mock.MagicMock(distributed=fake_dist)
    with mock.patch.dict(
        "sys.modules",
        {"torch": fake_torch, "torch.distributed": fake_dist},
    ):
      with self.assertRaisesRegex(ValueError, "not a member"):
        _get_process_index_and_count(group=object())


if __name__ == "__main__":
  absltest.main()
