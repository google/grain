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
"""Checks that OSS Grain Package works end-to-end with JAX."""

from absl.testing import absltest
import grain


class JaxImportTest(absltest.TestCase):

  def test_with_jax(self):
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    ds = grain.MapDataset.source(jnp.arange(10)).map(lambda x: x + 1)

    for _ in ds:
      pass


if __name__ == "__main__":
  absltest.main()
