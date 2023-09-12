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


if __name__ == '__main__':
  absltest.main()
