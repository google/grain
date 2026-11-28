/* Copyright 2022 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements random_index_shuffle() by using a simple block chiper
// for pseudorandom permutations.
// This idea is described as cycle-walking cipher in
// https://www.cs.ucdavis.edu/~rogaway/papers/subset.pdf
//
// We use the Simon block cipher described in
// https://eprint.iacr.org/2013/404
// and following recommendations in
// https://nsacyber.github.io/simon-speck/implementations/ImplementationGuide1.1.pdf.
// However we use a single fixed key size and support arbitrary block sizes.
// Further we fixed the number of rounds in the Feistel structure to be always
// 4. This reduces the computational cost and still gives good shuffle behavior.
//
// Warning: Given the modifications description above this implementation should
// not be used for application that require cryptographic secure RNGs.

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

#include <assert.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

namespace grain {
namespace random {
namespace {

// Minimum block size (2 * word size). simon_f requires word size >= 8, and
// smaller block sizes give poor randomness.
constexpr int kMinBlockSize = 16;

// Returns the keys per round for the cipher.
// This is not the proposed key schedule for a Simon cipher but simply uses
// std::seed_seq. We found that this gives better results.
std::vector<uint32_t> generate_keys(const uint32_t seed,
                                    const uint32_t rounds) {
  std::vector<uint32_t> rk(rounds);
  std::seed_seq seq{seed};
  seq.generate(rk.begin(), rk.end());
  return rk;
}

template <int W>
uint32_t simon_f(const uint32_t x) {
  static_assert(W >= 8 && W <= 32, "Word size must be in [8, 32] bits.");
  const uint32_t rotl_1 = (x << 1) | (x >> (W - 1));
  const uint32_t rotl_8 = (x << 8) | (x >> (W - 8));
  const uint32_t rotl_2 = (x << 2) | (x >> (W - 2));
  return (rotl_1 & rotl_8) ^ rotl_2;
}

// Encrypts the given value using the Simon cipher.
// This should not be used to encrypt data. It is not secure. We only use it
// to generate pseudorandom permutations.
template <int W>
uint64_t simon_encrypt(const uint64_t value, const uint32_t* round_keys,
                       const uint32_t rounds) {
  static_assert(W >= 8 && W <= 32, "Word size must be in [8, 32] bits.");
  constexpr uint32_t kWordMask = static_cast<uint32_t>((uint64_t{1} << W) - 1);
  uint32_t left = static_cast<uint32_t>(value >> W) & kWordMask;
  uint32_t right = static_cast<uint32_t>(value) & kWordMask;
  for (uint32_t i = 0; i < rounds; i += 2) {
    left = (left ^ simon_f<W>(right) ^ round_keys[i]) & kWordMask;
    right = (right ^ simon_f<W>(left) ^ round_keys[i + 1]) & kWordMask;
  }
  return (static_cast<uint64_t>(left) << W) | static_cast<uint64_t>(right);
}

// In the original implementation the number of rounds depends on the block
// size, key size and key words. For our purposes of random shuffle a 4 to 8
// rounds is enough.
// W = word size
// B = 2 * W = block size
template <int B>
uint64_t shuffle_element(const uint64_t index, const uint64_t max_index,
                         const uint32_t* round_keys, const uint32_t rounds) {
  uint64_t new_index = index;
  while (true) {
    new_index = simon_encrypt<B / 2>(new_index, round_keys, rounds);
    if (new_index <= max_index) {
      return new_index;
    }
  }
}

int compute_block_size(const uint64_t max_index) {
  // Block size must be large enough to represent max_index and even (since
  // word size is half of it). We force at least 16 bits as minimum block size
  // since we observed pattern in the permutations below.
  int block_size = static_cast<int>(std::ceil(std::log2(max_index)));
  block_size = std::max(block_size + block_size % 2, kMinBlockSize);
  assert(block_size > 0 && block_size % 2 == 0 && block_size <= 64);
  return block_size;
}

template <typename Fn>
decltype(auto) dispatch_block_size(const int block_size, Fn&& fn) {
#define HANDLE_BLOCK_SIZE(B) \
  case B:                    \
    return fn(std::integral_constant<int, B>{});

  switch (block_size) {
    HANDLE_BLOCK_SIZE(16);
    HANDLE_BLOCK_SIZE(18);
    HANDLE_BLOCK_SIZE(20);
    HANDLE_BLOCK_SIZE(22);
    HANDLE_BLOCK_SIZE(24);
    HANDLE_BLOCK_SIZE(26);
    HANDLE_BLOCK_SIZE(28);
    HANDLE_BLOCK_SIZE(30);
    HANDLE_BLOCK_SIZE(32);
    HANDLE_BLOCK_SIZE(34);
    HANDLE_BLOCK_SIZE(36);
    HANDLE_BLOCK_SIZE(38);
    HANDLE_BLOCK_SIZE(40);
    HANDLE_BLOCK_SIZE(42);
    HANDLE_BLOCK_SIZE(44);
    HANDLE_BLOCK_SIZE(46);
    HANDLE_BLOCK_SIZE(48);
    HANDLE_BLOCK_SIZE(50);
    HANDLE_BLOCK_SIZE(52);
    HANDLE_BLOCK_SIZE(54);
    HANDLE_BLOCK_SIZE(56);
    HANDLE_BLOCK_SIZE(58);
    HANDLE_BLOCK_SIZE(60);
    HANDLE_BLOCK_SIZE(62);
    default:
      return fn(std::integral_constant<int, 64>{});
  }
#undef HANDLE_BLOCK_SIZE
}

}  // namespace

uint64_t index_shuffle(const uint64_t index, const uint64_t max_index,
                       const uint32_t seed, const uint32_t rounds) {
  // rounds should be in [4, 1024] and number of rounds must be even.
  assert(rounds >= 4 && rounds <= 1024 && rounds % 2 == 0);
  // Assert the index is bounded by [0, max_index].
  assert(index <= max_index);
  if (max_index == 0) {
    return 0;
  }
  const int block_size = compute_block_size(max_index);
  const std::vector<uint32_t> round_keys = generate_keys(seed, rounds);
  const uint32_t* keys_ptr = round_keys.data();
  return dispatch_block_size(block_size, [&](auto block_size_tag) {
    constexpr int kBlockSize = decltype(block_size_tag)::value;
    return shuffle_element<kBlockSize>(index, max_index, keys_ptr, rounds);
  });
}

std::vector<uint64_t> index_shuffle_range(const uint64_t start,
                                          const uint64_t stop,
                                          const uint64_t max_index,
                                          const uint32_t seed,
                                          const uint32_t rounds) {
  // rounds should be in [4, 1024] and number of rounds must be even.
  assert(rounds >= 4 && rounds <= 1024 && rounds % 2 == 0);
  if (start >= stop) {
    return {};
  }
  assert(stop - 1 <= max_index);
  const uint64_t count = stop - start;
  assert(count <= std::numeric_limits<size_t>::max());
  if (max_index == 0) {
    return std::vector<uint64_t>(static_cast<size_t>(count), 0);
  }
  const int block_size = compute_block_size(max_index);
  const std::vector<uint32_t> round_keys = generate_keys(seed, rounds);
  const uint32_t* keys_ptr = round_keys.data();
  return dispatch_block_size(block_size, [&](auto block_size_tag) {
    constexpr int kBlockSize = decltype(block_size_tag)::value;
    std::vector<uint64_t> shuffled(static_cast<size_t>(count));
    for (size_t offset = 0; offset < shuffled.size(); ++offset) {
      shuffled[offset] = shuffle_element<kBlockSize>(start + offset, max_index,
                                                     keys_ptr, rounds);
    }
    return shuffled;
  });
}

}  // namespace random
}  // namespace grain
