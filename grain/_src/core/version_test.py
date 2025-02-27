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
"""Checks that tests in OSS are run with the correct version of Python."""
# Make sure grain can be imported.
from grain import python as grain  # pylint: disable=unused-import

import os
import sys
from absl.testing import absltest


class VersionTest(absltest.TestCase):

  def test_python_version(self):
    expected = os.getenv("PYTHON_VERSION")
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != expected:
      raise ValueError(
          f"expected version '{expected}' is different than returned"
          f" '{current}'"
      )


if __name__ == "__main__":
  absltest.main()
