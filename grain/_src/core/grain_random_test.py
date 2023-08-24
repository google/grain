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
"""Unit tests for the random module."""
import contextlib
import itertools

from absl.testing import parameterized
from grain._src.core import grain_random
import jax
import numpy as np
import tensorflow as tf


class RandomTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the random module."""

  @parameterized.parameters(
      {"seed": 585},
      {"seed": (5424, 95849)},
      {"seed": np.asarray([5424, 95849])},
      {"seed": np.int32(95849)},
      {"seed": np.int64(5424)},
  )
  def test_rng_for_jax(self, seed):
    seed = grain_random.as_rng_key(seed)
    # Verify we can split the seed.
    seed, _ = jax.random.split(seed)
    # Verify we can generate a random number.
    jax.random.uniform(seed, [])

  @parameterized.parameters(
      itertools.product(
          [585, (5424, 95849), np.asarray([5424, 95849])],
          [
              # No context.
              (None, None),
              # Each separately.
              ("custom", None),
              ("threefry2x32", None),
              ("rbg", None),
              # implementation frist, custom afterwards.
              ("threefry2x32", "custom"),
              ("rbg", "custom"),
              # custom first, implementation afterwards.
              ("custom", "threefry2x32"),
              ("custom", "rbg"),
          ],
      )
  )
  def test_rng_for_jax_custom(self, seed, contexts):
    def get_ctx(name):
      if name is None:
        return contextlib.nullcontext()
      if name == "threefry2x32":
        return jax.default_prng_impl("threefry2x32")
      if name == "rbg":
        return jax.default_prng_impl("rbg")
      if name == "custom":
        return jax.enable_custom_prng()
      assert False

    with get_ctx(contexts[0]):
      with get_ctx(contexts[1]):
        seed = grain_random.as_rng_key(seed)
        # Verify we can split the seed.
        seed, _ = jax.random.split(seed)
        # Verify we can generate a random number.
        jax.random.uniform(seed, [])


if __name__ == "__main__":
  tf.test.main()
