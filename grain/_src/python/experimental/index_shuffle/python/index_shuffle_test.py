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

from concurrent import futures
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.experimental.index_shuffle.python import index_shuffle_module as index_shuffle


class IndexShuffleTest(parameterized.TestCase):

  def test_index_shuffle(self):
    max_index = 46_204
    seen = set()
    for x in range(max_index + 1):
      y = index_shuffle.index_shuffle(x, max_index, seed=52, rounds=4)
      self.assertBetween(y, 0, max_index)
      seen.add(y)
    self.assertLen(seen, max_index + 1)

  def test_index_shuffle_huge_number(self):
    max_index = 1_234_567_891
    seen = set()
    for x in range(10_000):
      y = index_shuffle.index_shuffle(x, max_index, seed=27, rounds=4)
      self.assertBetween(y, 0, max_index)
      seen.add(y)
    self.assertLen(seen, 10_000)

  def test_index_shuffle_single_record(self):
    self.assertEqual(
        0, index_shuffle.index_shuffle(index=0, max_index=0, seed=0, rounds=4)
    )

  def test_index_shuffle_invalid_rounds(self):
    regex = r'rounds must be an even integer between 4 and 1024'
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(index=0, max_index=8, seed=33, rounds=2)
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(index=0, max_index=8, seed=33, rounds=1025)
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(index=0, max_index=8, seed=76, rounds=5)

  def test_index_shuffle_rounds_too_large(self):
    regex = r'rounds must be an even integer between 4 and 1024'
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(
          index=0, max_index=1, seed=0, rounds=4294967294
      )

  def test_index_shuffle_invalid_index(self):
    regex = r'index must be in \[0, max_index\]'
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(index=-1, max_index=8, seed=33, rounds=4)
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle(index=9, max_index=8, seed=76, rounds=4)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_record',
          start=0,
          stop=1,
          max_index=0,
          seed=0,
          rounds=4,
      ),
      dict(
          testcase_name='max_index_1',
          start=0,
          stop=2,
          max_index=1,
          seed=0,
          rounds=8,
      ),
      dict(
          testcase_name='max_index_2',
          start=0,
          stop=3,
          max_index=2,
          seed=7,
          rounds=8,
      ),
      dict(
          testcase_name='max_index_3',
          start=0,
          stop=4,
          max_index=3,
          seed=2**32 - 1,
          rounds=8,
      ),
      dict(
          testcase_name='power_of_two_minus_one',
          start=65_500,
          stop=65_536,
          max_index=2**16 - 1,
          seed=42,
          rounds=8,
      ),
      dict(
          testcase_name='power_of_two',
          start=0,
          stop=64,
          max_index=2**16,
          seed=42,
          rounds=8,
      ),
      dict(
          testcase_name='power_of_two_plus_one',
          start=65_500,
          stop=65_538,
          max_index=2**16 + 1,
          seed=42,
          rounds=8,
      ),
      dict(
          testcase_name='above_uint32_boundary',
          start=2**32 - 10,
          stop=2**32 + 10,
          max_index=2**32 + 12_345,
          seed=2**32 - 1,
          rounds=8,
      ),
      dict(
          testcase_name='int64_max',
          start=2**63 - 9,
          stop=2**63 - 1,
          max_index=2**63 - 1,
          seed=123,
          rounds=1024,
      ),
  )
  def test_index_shuffle_range_matches_scalar(
      self, start: int, stop: int, max_index: int, seed: int, rounds: int
  ):
    actual = index_shuffle.index_shuffle_range(
        start=start, stop=stop, max_index=max_index, seed=seed, rounds=rounds
    )
    expected = [
        index_shuffle.index_shuffle(i, max_index, seed=seed, rounds=rounds)
        for i in range(start, stop)
    ]
    self.assertEqual(actual, expected)
    for value in actual:
      self.assertIs(type(value), int)

  @parameterized.named_parameters(
      dict(testcase_name='zero_empty', start=0, stop=0, max_index=0),
      dict(testcase_name='end_empty', start=1, stop=1, max_index=0),
      dict(testcase_name='interior_empty', start=5, stop=5, max_index=8),
      dict(testcase_name='reversed', start=7, stop=3, max_index=8),
      dict(testcase_name='past_end_empty', start=12, stop=12, max_index=8),
  )
  def test_index_shuffle_range_empty(
      self, start: int, stop: int, max_index: int
  ):
    self.assertEmpty(
        index_shuffle.index_shuffle_range(
            start=start, stop=stop, max_index=max_index, seed=0, rounds=8
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='rounds_too_small',
          start=0,
          stop=1,
          max_index=8,
          rounds=2,
          regex=r'rounds must be an even integer between 4 and 1024',
      ),
      dict(
          testcase_name='rounds_odd',
          start=0,
          stop=0,
          max_index=8,
          rounds=5,
          regex=r'rounds must be an even integer between 4 and 1024',
      ),
      dict(
          testcase_name='rounds_too_large',
          start=0,
          stop=1,
          max_index=8,
          rounds=1026,
          regex=r'rounds must be an even integer between 4 and 1024',
      ),
      dict(
          testcase_name='negative_max_index',
          start=0,
          stop=0,
          max_index=-1,
          rounds=4,
          regex=r'max_index must be non-negative',
      ),
      dict(
          testcase_name='negative_start',
          start=-1,
          stop=2,
          max_index=8,
          rounds=4,
          regex=r'start and stop must be non-negative',
      ),
      dict(
          testcase_name='negative_empty_range',
          start=-2,
          stop=-2,
          max_index=8,
          rounds=4,
          regex=r'start and stop must be non-negative',
      ),
      dict(
          testcase_name='negative_stop',
          start=2,
          stop=-1,
          max_index=8,
          rounds=4,
          regex=r'start and stop must be non-negative',
      ),
      dict(
          testcase_name='stop_past_max_index',
          start=0,
          stop=10,
          max_index=8,
          rounds=4,
          regex=r'\[start, stop\) must be within \[0, max_index \+ 1\]',
      ),
      dict(
          testcase_name='stop_past_zero_max_index',
          start=0,
          stop=2,
          max_index=0,
          rounds=4,
          regex=r'\[start, stop\) must be within \[0, max_index \+ 1\]',
      ),
  )
  def test_index_shuffle_range_invalid_args(
      self, start: int, stop: int, max_index: int, rounds: int, regex: str
  ):
    with self.assertRaisesRegex(ValueError, regex):
      index_shuffle.index_shuffle_range(
          start=start, stop=stop, max_index=max_index, seed=33, rounds=rounds
      )

  def test_index_shuffle_range_threaded(self):
    max_index = 99_999
    chunk_size = 25_000
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
      chunks = list(
          executor.map(
              lambda s: index_shuffle.index_shuffle_range(
                  s, s + chunk_size, max_index, seed=99, rounds=8
              ),
              range(0, max_index + 1, chunk_size),
          )
      )
    combined = list(itertools.chain.from_iterable(chunks))
    self.assertEqual(
        combined,
        index_shuffle.index_shuffle_range(
            0, max_index + 1, max_index, seed=99, rounds=8
        ),
    )
    self.assertLen(set(combined), max_index + 1)


if __name__ == '__main__':
  absltest.main()
