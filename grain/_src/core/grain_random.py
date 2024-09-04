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
"""Type definition and methods for handling random numbers.

Grain uses counter-based pseudorandom number generators (PRNGs) for fast and
reproducible random numbers.
This means that there is no global random seed or a global stateful PRNGs
object. Instead, random seeds are always passed explicitly and users should
understand the rules outlined below (this applies both to the internals of
Grain as well as user defined preprocessing and extensions).

## Motivation
We want Grain input pipelines to be reproducible while still being highly
parallelizable. This very much overlaps with the design choices for JAX's PRNG
([design notes](github.com/google/jax/blob/main/docs/design_notes/prng.md)) and
their choice for counter-based PRNGs like ThreeFry and Philox.
NumPy, JAX and TF all contain implementations of ThreeFry/Philox.

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
- Never use the `random` module from the Python standard library.
- Either use methods in `jax.random`, random methods in TensorFlow that start
  with `stateless_` (e.g. `tf.random.stateless_uniform()`) or non-global
  NumPy random generators.
- Do not use the global RNG in `np.random`. Instead create a new generator
  with appropiate bit generator: `np.random.default_rng(seed)` or
  `np.random.Generator(np.random.Philox(seed))` and call random functions on it.
  The generator (or more accurately the underlying bit generator) is stateful
  and should not be used across between threads.
- If you want to use a new random seed each time your program starts, you can
  use `int.from_bytes(os.urandom(4), sys.byteorder)` and log the value.
- Do not re-use a random seed. Instead, split the random seed using
  `jax.random.split()` or `tf.random.experimental.stateless_split()`

## JAX
JAX supports multiple RNG implementations. Since Grain operations run on CPU,
we recommend the default (ThreeFry). It's stable across JAX/XLA versions.
Internally, ThreeFry uses a uint32 type vector of shape [2] as state.

## NumPy
NumPy contains several suitable RNG implementations (aka bit generators). We
recommend using `np.random.Philox` which is very similar to the ThreeFry
implementation in JAX and TF. However `PCG64` (default as of writting) and
`PCG64DXSM` are also suitable for input pipelines. Just don't use the global
RNG and always construct your own `np.random.Generator`.

## TensorFlow
TensorFlow implements multiple counter-based RNGs (ThreeFry and Philox) but
doesn't provide any guidance on which one to use. We recommend to not set the
algorithm explicitly and let TF select the RNG (on CPUs it seems to select
Philox). You can derive a random seed for stateless random functions using
`tf.random.Generator.from_seed(my_integer).make_seeds(1)[:, 0]`.
Splitting seeds can be done using:
```
seed, seed_for_my_op = tf.unstack(tf.random.experimental.stateless_split(seed))
```
"""
import os
import sys
from typing import Optional, Sequence, Union

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

RNGKey = jax.Array
RNGKeyLike = Union[RNGKey, int, Sequence[int], np.ndarray]


def as_rng_key(seed: RNGKeyLike) -> RNGKey:
  """Returns an RNGKey from the input.

  Args:
    seed: Rndom seed as single integer or tuple/array of 2 integers.

  Returns:
    RNGKey which is currently a NumPy array of dtype uint32 and shape [2].
  """
  if hasattr(seed, "dtype") and jax.dtypes.issubdtype(
      seed.dtype, jax.dtypes.prng_key
  ):
    return seed
  if isinstance(seed, (int, jnp.integer)):
    return jax.random.key(seed)
  if isinstance(seed, jnp.ndarray):
    return jax.random.wrap_key_data(seed)
  if len(seed) != 2:
    raise ValueError(
        "Random seed must be a single integer or a "
        f"Tuple[int, int] but got {seed!r}"
    )
  return jax.random.key(sum(seed))


def make_rng_key(seed: Optional[RNGKeyLike]) -> RNGKey:
  logging.error(
      "Using deprecated method make_rng_key(). Please always pass "
      "None or a single integer as random seed to Grain methods."
  )
  if seed is None:
    seed = int.from_bytes(os.urandom(4), sys.byteorder)
  return as_rng_key(seed)
