#include "grain/_src/cpp/transformations/packing/mffd.h"

#include <cstdint>
#include <vector>

#include "testing/base/public/gunit.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/strings/str_format.h"
#include "util/gtl/iterator_adaptors.h"

namespace grain {

void PrintStats(const std::vector<Bin>& bins, int32_t capacity) {
  int padding = 0, non_padding = 0;
  for (const auto& bin : bins) {
    padding += bin.remaining_capacity;
    non_padding += capacity - bin.remaining_capacity;
  }
  LOG(INFO) << absl::StrFormat(
      "Total useful: %d, total padding: %d, useful ratio: %.2f ", non_padding,
      padding, (100.0f * non_padding / (non_padding + padding)));
  LOG(INFO) << absl::StrFormat(
      "Total useful without last batch: %.2f ",
      100.0f * (non_padding - (capacity - bins.back().remaining_capacity)) /
          ((non_padding - (capacity - bins.back().remaining_capacity) +
            padding - bins.back().remaining_capacity)));
}

TEST(Mffd, LargeInput) {
  auto capacity = 2 << 15;
  for (int i = 0; i < 42; ++i) {
    std::vector<Item> input;
    absl::BitGen gen;
    for (int i = 0; i < 20000; ++i) {
      input.push_back(Item{.id = i, .weight = absl::Uniform(gen, 0, 32000)});
    }
    PrintStats(ModifiedFirstFitDescendingBinPacking(input, capacity), capacity);
  }
}

TEST(Mffd, SmallInputs) {
  struct TestSpec {
    std::vector<int32_t> in;
    int32_t cap;
    int32_t want;
  };
  for (const auto& t : std::vector<TestSpec>{
           {.in = {4, 8, 1, 4, 2, 1}, .cap = 10, .want = 2},
           {.in = {9, 8, 2, 2, 5, 4}, .cap = 10, .want = 4},
           {.in = {2, 5, 4, 7, 1, 3, 8}, .cap = 10, .want = 3},
           {.in = {48, 30, 19, 36, 36, 27, 42, 42, 36, 24, 30},
           .cap = 100,
           .want = 4},
           {.in = {5, 6, 4, 2, 10, 3}, .cap = 10, .want = 3},
           {.in = {50, 3, 48, 53, 53, 4, 3, 41, 23, 20, 52, 49},
           .cap = 100,
           .want = 5},
           {.in = {99, 93, 90, 88, 80, 10, 10, 6, 5, 5, 4, 4},
           .cap = 100,
           .want = 5}}) {
    std::vector<Item> input;
    for (int i = 0; i < t.in.size(); ++i) {
      input.push_back(Item{.id = i, .weight = t.in[i]});
    }
    // Pack items in bins.
    auto bins = ModifiedFirstFitDescendingBinPacking(input, t.cap);
    EXPECT_EQ(t.want, bins.size());

    // Check that all items in the input are in some bin.
    {
      // Count up all items of each weight size.
      absl::flat_hash_map<int32_t, int32_t> weight_counts;
      for (const auto& input : t.in) {
        ++weight_counts[input];
      }
      // And then count down for those present in bins.
      for (const auto& bin : bins) {
        for (const auto& item : bin.items) {
          --weight_counts[item.weight];
        }
      }
      // Finally all counts should be zero if all items in input were present in
      // bins.
      for (const auto& weight : gtl::value_view(weight_counts)) {
        EXPECT_EQ(0, weight);
      }
    }
  }
}

}  // namespace grain
