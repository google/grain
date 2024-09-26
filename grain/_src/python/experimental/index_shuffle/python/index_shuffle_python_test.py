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
"""Minimal unit test for the Python wrapper of index_shuffle."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.experimental.index_shuffle.python import index_shuffle_python


class IndexShuffleTest(absltest.TestCase):

  def test_index_shuffle(self):
    max_index = 46_204
    seen = set()
    for x in range(max_index + 1):
      y = index_shuffle_python.index_shuffle(x, max_index, seed=52, rounds=4)
      self.assertBetween(y, 0, max_index)
      seen.add(y)
    self.assertLen(seen, max_index + 1)

  def test_index_shuffle_huge_number(self):
    max_index = 1_234_567_891
    seen = set()
    for x in range(10_000):
      y = index_shuffle_python.index_shuffle(x, max_index, seed=27, rounds=4)
      self.assertBetween(y, 0, max_index)
      seen.add(y)
    self.assertLen(seen, 10_000)

  def test_index_shuffle_single_record(self):
    self.assertEqual(
        0,
        index_shuffle_python.index_shuffle(
            index=0, max_index=0, seed=0, rounds=4
        ),
    )


class IndexShuffleStabilityTest(parameterized.TestCase):

  @parameterized.product(seed=[0, 1, 113, 299_792_458], rounds=[1, 2, 3, 4])
  def test_basic_stability(self, seed, rounds):
    seen = {}
    for i in range(100):
      seen[i] = index_shuffle_python.index_shuffle(i, 100, seed=seed, rounds=4)
    # Here we also make sure that `rounds` is ignored, as documented.
    for i in range(100):
      self.assertEqual(
          seen[i],
          index_shuffle_python.index_shuffle(i, 100, seed=seed, rounds=rounds),
      )

  @parameterized.product(max_index=[10, 20], new_elements=[1, 3])
  def test_advanced_stability(self, max_index, new_elements):
    # This test is focused on the scenario where we run index_shuffle on a
    # dataset of size N and then add a few more elements to assess the impact.
    #
    # Ideally this would result in a permutation that is largely similar to the
    # original one so that we compare apples to apples.
    #
    # Unfortunately, this is not the case. The following test shows that the
    # diff between the original and the new permutation is essentially
    # completely random. In other words, there is no stability w.r.t. the size
    # of the dataset (which means ablation studies where data is removed or
    # added will be inherently noisy due to data ordering effects).
    seed = 123
    seen = {}
    for i in range(max_index):
      seen[i] = index_shuffle_python.index_shuffle(
          i, max_index, seed=seed, rounds=4
      )
    for i in range(max_index + new_elements):
      if i in seen:
        diff = abs(
            seen[i]
            - index_shuffle_python.index_shuffle(
                new_elements,
                max_index + new_elements,
                seed=seed,
                rounds=4,
            )
        )
        # Unfortunately, the following would fail as mentioned above.
        # self.assertLess(abs(diff), new_elements)

        # The following works simply because the range is limited
        # correctly to the max index..
        self.assertLess(diff, max_index + new_elements)


if __name__ == "__main__":
  absltest.main()
