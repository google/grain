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
"""Checks that OSS Grain Package works end-to-end with TF."""

import os

from absl.testing import absltest
import grain


@absltest.skipIf(
    os.getenv("PYTHON_MINOR_VERSION") >= "13",
    "TF is not available on Python 3.13 and above.",
)
class TFImportTest(absltest.TestCase):

  def test_with_tf(self):
    import tensorflow as tf  # pylint: disable=g-import-not-at-top

    ds = grain.MapDataset.source(range(10)).map(tf.convert_to_tensor)

    for _ in ds:
      pass


if __name__ == "__main__":
  absltest.main()
