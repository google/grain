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

from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import grain
from grain._src.core import transforms
from grain._src.python.dataset import dataset
import jax


class _AddJaxRandomInt(transforms.JaxStatelessRandomMap):

  def stateless_random_map(self, element: int, key: Any) -> int:
    return element + int(jax.random.randint(key, (), 0, 100))


class JaxStatelessRandomMapTest(parameterized.TestCase):

  def test_docstring_example(self):
    map_ds = dataset.MapDataset.range(5).seed(42)
    map_ds = map_ds.stateless_random_map(
        lambda x, key: x + int(jax.random.randint(key, (), 5, 10))
    )
    self.assertTrue(set(map_ds).issubset(set(range(5, 15))))

    iter_ds = dataset.MapDataset.range(5).to_iter_dataset().seed(42)
    iter_ds = iter_ds.stateless_random_map(
        lambda x, key: x + int(jax.random.randint(key, (), 5, 10))
    )
    self.assertTrue(set(iter_ds).issubset(set(range(5, 15))))

  def test_default_callable_wraps_in_jax_stateless_random_map(self):
    fn = lambda x, key: x + int(jax.random.randint(key, (), 0, 100))
    direct_ds = dataset.MapDataset.range(6).seed(42).stateless_random_map(fn)
    explicit_ds = (
        dataset.MapDataset.range(6)
        .seed(42)
        .stateless_random_map(grain.transforms.JaxStatelessRandomMap(fn))
    )
    self.assertIsInstance(
        grain.transforms.JaxStatelessRandomMap(fn),
        transforms.StatelessRandomMap,
    )
    self.assertEqual(list(direct_ds), list(explicit_ds))
    self.assertEqual(direct_ds[3], direct_ds[3])

  def test_iter_dataset_and_apply_support(self):
    fn = lambda x, key: x + int(jax.random.randint(key, (), 0, 100))
    map_vals = list(
        dataset.MapDataset.range(5).stateless_random_map(fn, seed=7)
    )
    iter_vals = list(
        dataset.MapDataset.range(5)
        .to_iter_dataset()
        .stateless_random_map(fn, seed=7)
    )
    self.assertEqual(iter_vals, map_vals)

    applied = dataset.MapDataset.range(5).seed(7).apply(_AddJaxRandomInt())
    seeded_direct = dataset.MapDataset.range(5).seed(7).stateless_random_map(fn)
    self.assertEqual(list(applied), list(seeded_direct))

  def test_supports_64bit_seeds_and_large_indices(self):
    transform = transforms.JaxStatelessRandomMap(
        lambda x, key: int(jax.random.randint(key, (), 0, 1000))
    )
    key_fn_a = transform.make_key_for_index(2**64 - 1)
    key_fn_b = transform.make_key_for_index(2**64 - 2)
    self.assertNotEqual(
        int(jax.random.bits(key_fn_a(0))),
        int(jax.random.bits(key_fn_b(0))),
    )
    # Distinct multi-epoch indices exceeding 2**32 produce distinct keys.
    self.assertNotEqual(
        int(jax.random.bits(key_fn_a(5))),
        int(jax.random.bits(key_fn_a((1 << 32) + 5))),
    )

  def test_rejects_stateful_random_map_or_missing_override(self):
    class StatefulAdd(transforms.RandomMap):

      def random_map(self, element: int, rng) -> int:
        return element + int(rng.integers(0, 10))

    with self.assertRaisesRegex(TypeError, "JaxStatelessRandomMap"):
      transforms.JaxStatelessRandomMap(StatefulAdd())  # pyrefly: ignore[bad-argument-type]

    empty = transforms.JaxStatelessRandomMap()
    key_fn = empty.make_key_for_index(0)
    with self.assertRaisesRegex(NotImplementedError, "JaxStatelessRandomMap"):
      empty.stateless_random_map(1, key_fn(0))


if __name__ == "__main__":
  absltest.main()
