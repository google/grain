#include "grain/_src/cpp/transformations/packing/mffd.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "third_party/absl/algorithm/container.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"

namespace grain {

bool Item::operator<(const Item& other) const { return weight < other.weight; }
bool Item::operator>(const Item& other) const { return weight > other.weight; }

bool Bin::Fits(const Item& item) { return item.weight <= remaining_capacity; }

absl::Status Bin::Place(const Item& item) {
  if (item.weight > remaining_capacity) {
    return absl::InvalidArgumentError(
        absl::StrCat("capacity exceeded: capacity is ", remaining_capacity,
                     ", item is ", item.weight));
  }
  remaining_capacity -= item.weight;
  items.push_back(item);
  return absl::OkStatus();
}

namespace {

// List containing all items sorted by descending weight.
//
// Items are categorized in 4 categories by size:
//  - large, size > 1/2 capacity,
//  - medium, 1/3 capacity < size < 1/2 capacity,
//  - small, 1/6 capacity < size < 1/3 capacity,
//  - tiny, 1/6 capacity > size
struct WeightList {
  WeightList(const std::vector<Item>& in, int capacity) : items(in) {
    // Sort the items descendingly.
    absl::c_stable_sort(items, std::greater<Item>());

    // Calculate pointers to begin/end item in each category.
    for (int i = 0; i < items.size(); ++i) {
      if (items[i].weight > capacity / 2) {
        if (large.first == -1) {
          large.first = i;
        }
        if (i < items.size() - 1 && items[i + 1].weight <= capacity / 2 &&
            large.second == -1) {
          large.second = i;
        }
      } else if (items[i].weight > capacity / 3) {
        if (medium.first == -1) {
          medium.first = i;
        }
        if (i < items.size() - 1 && items[i + 1].weight <= capacity / 3 &&
            medium.second == -1) {
          medium.second = i;
        }
      } else if (items[i].weight > capacity / 6) {
        if (small.first == -1) {
          small.first = i;
          small.second = i;
        }
        if (i < items.size() - 1 && items[i + 1].weight <= capacity / 6 &&
            small.second == -1) {
          small.second = i;
        }
      } else {
        if (tiny.first == -1) {
          tiny.first = i;
        }
      }
    }
    if (tiny.first != -1) {
      tiny.second = items.size() - 1;
    }
  }

  // Whether the item under `index` is assigned to any bins.
  bool IsAssigned(int32_t index) { return assigned.contains(index); }

  // Pointers to the indices of each category.
  std::pair<int, int> large = {-1, -1};
  std::pair<int, int> medium = {-1, -1};
  std::pair<int, int> small = {-1, -1};
  std::pair<int, int> tiny = {-1, -1};

  // Items in the weight list, descending by weight.
  std::vector<Item> items;

  // Indices of items that are already assigned to a bin.
  absl::flat_hash_set<int32_t> assigned;
};

void Place(Bin& bin, WeightList& wl, int32_t index) {
  if (auto status = bin.Place(wl.items[index]); !status.ok()) {
    LOG(FATAL) << "Cannot fit item in a bin that has sufficient "
                  "capacity; should never happen: ";
  }
  wl.assigned.insert(index);
}

void FirstFitDescending(WeightList& wl, int capacity, std::vector<Bin>& bins) {
  for (int i = 0; i < wl.items.size(); ++i) {
    if (wl.IsAssigned(i)) {
      continue;
    }
    bool placed = false;

    // Try to place the current item in existing bins.
    for (auto& bin : bins) {
      if (bin.Fits(wl.items[i])) {
        Place(bin, wl, i);
        placed = true;
        break;
      }
    }

    // If the item couldn't fit in any existing bin, create a new bin.
    if (!placed) {
      Bin bin = {.remaining_capacity = capacity};
      Place(bin, wl, i);
      bins.push_back(bin);
    }
  }
}

}  // namespace

std::vector<Bin> ModifiedFirstFitDescendingBinPacking(
    const std::vector<Item>& items, int capacity) {
  // Sort item sizes in descending order.
  WeightList wl(items, capacity);

  std::vector<Bin> bins;

  // Allot a bin for each large item, ordered largest to smallest.
  for (int i = wl.large.first; i <= wl.large.second; ++i) {
    if (i < 0) {
      continue;
    }
    Bin bin = {.remaining_capacity = capacity};
    Place(bin, wl, i);
    bins.push_back(bin);
    ++wl.large.first;
  }

  // Proceed forward through the bins. On each: If the smallest remaining medium
  // item does not fit, skip this bin. Otherwise, place the largest remaining
  // medium item that fits.
  for (auto& bin : bins) {
    if (wl.medium.second > bin.remaining_capacity) {
      continue;
    }
    for (int i = wl.medium.first; i <= wl.medium.second; ++i) {
      if (i < 0 || wl.IsAssigned(i)) {
        continue;
      }
      if (bin.Fits(wl.items[i])) {
        Place(bin, wl, i);
        ++wl.medium.first;
        bin.has_medium = true;
        break;
      }
    }
  }

  // Proceed backward through those bins that do not contain a medium item. On
  // each: If the two smallest remaining small items do not fit, skip this bin.
  // Otherwise, place the smallest remaining small item and the largest
  // remaining small item that fits.
  for (int i = bins.size() - 1; i >= 0; --i) {
    if (bins[i].has_medium) {
      continue;
    }
    if (wl.small.first != -1 && wl.small.second >= wl.small.first + 1 &&
        (wl.items[wl.small.second].weight +
             wl.items[wl.small.second - 1].weight >
         bins[i].remaining_capacity)) {
      continue;
    }
    if (wl.small.second - wl.small.first < 2) {
      continue;
    }
    Place(bins[i], wl, wl.small.second);
    --wl.small.second;
    for (int j = wl.small.first; j != wl.small.second; ++j) {
      if (bins[i].Fits(wl.items[j]) && !wl.assigned.contains(j)) {
        Place(bins[i], wl, j);
        break;
      }
    }
  }

  //  Proceed forward through all bins. If the smallest remaining item of any
  //  size class does not fit, skip this bin. Otherwise, place the largest
  //  item that fits and stay on this bin. Use FFD to pack the remaining items
  //  into new bins.
  for (auto& bin : bins) {
    for (int i = 0; i < wl.items.size(); ++i) {
      if (wl.assigned.contains(i)) {
        continue;
      }
      if (wl.items[i].weight <= bin.remaining_capacity) {
        Place(bin, wl, i);
      }
    }
  }

  // Place the rest of the items using the first-fit-descending algorithm.
  FirstFitDescending(wl, capacity, bins);
  return bins;
}

}  // namespace grain
