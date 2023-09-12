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
"""Pure Python version of `index_shuffle`.

This is roughly 10x slower than the C++ index_shuffle but still sufficiently
fast for many use cases. Use it if the C++ version (and it's CLIF wrapper) don't
work for you.
"""

import hashlib


def _fingerprint(*args) -> int:
  """A 128-bit fingerprint based on md5.

  For data shuffling - not for cryptography.

  Args:
    *args: any argument list that can be converted to a string

  Returns:
    an integer in [0, 2 ** 128)
  """
  return int.from_bytes(hashlib.md5(str(args).encode()).digest(), "little")


def index_shuffle(index: int, max_index: int, seed: int, rounds: int) -> int:
  """computes the position of `index` after a pseudorandom permutation on `[0, max_index])`.

  Based on Feistel ciphers.

  For data shuffling - not for cryptography.

  if i != j, then
  pseudorandom_permutation(n, i, seed) != pseudorandom_permutation(n, j, seed)

  Args:
    index: an integer in [0, max_index)
    max_index: A positive integer.
    seed: A posivtive integer used as seed for the pseudorandom permutation.
    rounds: Ignored. For compatibility with C++ version.

  Returns:
    An integer in [0, max_index].
  """
  del rounds
  if not isinstance(max_index, int):
    raise ValueError("n must be an integer")

  if index < 0 or index > max_index:
    raise ValueError("out of range")

  if max_index == 1:
    return 0

  # smallest k such that max_index fits in 2k bits
  k = (max_index.bit_length() + 1) // 2
  assert max_index <= 4**k
  # Permute repeatedly in [max_index, 4 ** k) until you land back in
  # [0, max_index]. This constitutes a permutation of [0, max_index].
  while True:
    # Feistel ciper on 2k bits - i.e. a permutation of [0, 4 ** k)
    a, b = index // (2**k), index % (2**k)
    for r in range(3):
      a, b = b, a ^ (_fingerprint(b, r, seed) % (2**k))
    index = a * (2**k) + b
    if index <= max_index:
      return int(index)
