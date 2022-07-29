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
"""Type definition and methods for handling random numbers.

Grain uses counter-based pseudorandom number generators (PRNGs) for fast and
reproducible random numbers.
This means that there is no random seed or a stateful PRNGs object. Instead,
random seeds are always passed explicitly and users should understand the
rules outlined below (this applies both to the internals of Grain as well as
user defined preprocessing and extensions).

## Motivation
We want Grain input pipelines to be reproducible while still being highly
parallelizable. This very much overlaps with the design choices for JAX's PRNG
([design notes](github.com/google/jax/blob/main/docs/design_notes/prng.md)).
Hence, we use `jax.random` where possible and equivalents otherwise (e.g.
stateless random ops in TF).

## General Rules
If you are not familiar with `jax.random`, we recommend reading
[the JAX
documentation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers)
or
[this
Colab](https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/05-random-numbers.ipynb).
to understand the concepts of counter-based PRNGs. You should be comfortable
with splitting your random seeds.

After that, just follow these rules:
- Do not use the Python `random` module or `numpy.random`. Both use a global
state.
- Either use methods in `jax.random` or random methods in TensorFlow that start
  with `stateless_` (e.g. `tf.random.stateless_uniform()`).
- If you want to use a new random seed each time your program starts, you can
use
  `int.from_bytes(os.urandom(4), sys.byteorder)` and log the value.
- Do not re-use a random seed. Instead, split the random seed using
  `jax.random.split()` or `tf.random.experimental.stateless_split()`

## JAX
JAX supports multiple RNG implementations. Since Grain operations run on CPU,
we recommend the default (ThreeFry). It's stable across JAX/XLA versions.
Internally, ThreeFry uses a uint32 type vector of shape [2].

## TensorFlow
TensorFlow implements multiple counter-based RNGs (ThreeFry and Pilox) but
doesn't provide any guidance on which one to use. We recommend to not set the
algorithm and simply pass a seed of dtype int32 (yes, not uint32!) and shape
[2].
Splitting seeds can be done using:
```
seed, seed_for_my_op = tf.unstack(tf.random.experimental.stateless_split(seed))
```

If you have a JAX RNG key, use `seed = tf.cast(seed, tf.int32)` before passing
it to TensorFlow functions.
"""

import os
import sys
from typing import Optional, Tuple, Union

import numpy as np

# Currently, Grain uses NumPy arrays of dtype uint32 and shape [2] for RNG keys.
# These are easy to create and can be used by TensorFlow by casting to tf.int32.
# In the future, we might switch to `jax.random.KeyArray`.
RNGKey = np.ndarray
RNGKeyLike = Union[int, Tuple[int, int], np.ndarray]


def as_rng_key(seed: RNGKeyLike) -> RNGKey:
  """Returns an RNGKey from the input.

  Args:
    seed: Rndom seed as single integer or tuple/array of 2 integers.

  Returns:
    RNGKey which is currently a NumPy array of dtype uint32 and shape [2].
  """
  if isinstance(seed, int):
    seed = [0, seed]
  if len(seed) != 2:
    raise ValueError('Random seed must be a single integer or a '
                     f'Tuple[int, int] but got {seed!r}')
  return np.asarray(seed, dtype=np.uint32)


def make_rng_key(seed: Optional[RNGKeyLike]) -> RNGKey:
  """Returns an RNGKey from the input or from random bits.

  Warning: If `seed` is None this will return a new value on each call. If you
  are running this in a distributed setting (e.g. jax.process_count() > 1) and
  you might want to use the same RNGKey for all processes. You can do this by
  broadcasting the RNGKey of process 0 to all other processes:
  from jax.experimental import multihost_utils
  seed = multihost_utils.broadcast_one_to_all(seed)

  Args:
    seed: Optional random seed. If None, `os.urandom()` is used to create a new
      seed.

  Returns:
    RNGKey which is currently a NumPy array of dtype uint32 and shape [2].
    If no seed or a single integer was provided as the seed, the first entry
    will be 0.
  """
  if seed is None:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)
  return as_rng_key(seed)
