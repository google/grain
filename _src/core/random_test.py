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
"""Unit tests for the random module."""

from absl.testing import parameterized
import grain._src.core.random as grain_random
import jax
import numpy as np
import tensorflow as tf


class RandomTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for the random module."""

  def test_no_seed_is_always_unique(self):
    for _ in range(100):
      self.assertNotAllEqual(
          grain_random.make_rng_key(None), grain_random.make_rng_key(None))

  @parameterized.parameters(
      {"seed": None},
      {"seed": 585},
      {"seed": (5424, 95849)},
      {"seed": np.asarray([5424, 95849])},
  )
  def test_rng_for_jax(self, seed):
    seed = grain_random.make_rng_key(seed)
    self.assertIsInstance(seed, grain_random.RNGKey)
    self.assertEqual(seed.dtype, np.uint32)
    # Is not a trival RNG key.
    self.assertNotAllEqual(seed, [0, 0])
    # Verify we can split the seed.
    seed, _ = jax.random.split(seed)
    # Verify we can generate a random number.
    jax.random.uniform(seed, [])

  @parameterized.parameters(
      {"seed": None},
      {"seed": 585},
      {"seed": (5424, 95849)},
      {"seed": np.asarray([5424, 95849])},
  )
  def test_rng_for_tf(self, seed):
    seed = grain_random.make_rng_key(seed)
    self.assertIsInstance(seed, grain_random.RNGKey)
    self.assertEqual(seed.dtype, np.uint32)
    # Is not a trival RNG key.
    self.assertNotAllEqual(seed, [0, 0])
    # Convert to TF.
    seed = tf.cast(seed, tf.int32)
    # Verify we can split the seed.
    seed, _ = tf.unstack(tf.random.experimental.stateless_split(seed))
    # Verify we can generate a random number.
    tf.random.stateless_uniform([], seed)


if __name__ == "__main__":
  tf.test.main()
