#ifndef THIRD_PARTY_PY_GRAIN__SRC_CPP_TRANSFORMATIONS_PACKING_MFFD_H_
#define THIRD_PARTY_PY_GRAIN__SRC_CPP_TRANSFORMATIONS_PACKING_MFFD_H_

#include <cstdint>
#include <vector>

#include "third_party/absl/status/status.h"
namespace grain {

// Represents an item in a bin. Its `weight` contributes to the bin's capacity.
struct Item {
  // Unique ID of the item; this is used for mapping back from item to actual
  // values.
  int32_t id;
  // Weight of the item it consumes of the bin's capacity.
  int32_t weight;

  bool operator<(const Item& other) const;
  bool operator>(const Item& other) const;
};

// A single bin that can fit a number of `Item`s with a given weight
// `remaining_capacity`.
struct Bin {
  // Places an `item` in the bin if it `Fits`.
  absl::Status Place(const Item& item);
  // Whether the `item` fits in the bin.
  bool Fits(const Item& item);

  // Bin's remaining capacity.
  int remaining_capacity;
  // `Item`s in the bin.
  std::vector<Item> items;
  // Whether the bin has a item of medium size (1/3 capacity < item size < 1/2
  // capacity) .
  bool has_medium = false;
};

// Packs all `items` in bins with given maximum `capacity`.
std::vector<Bin> ModifiedFirstFitDescendingBinPacking(
    const std::vector<Item>& items, int capacity);

}  // namespace grain

#endif  // THIRD_PARTY_PY_GRAIN__SRC_CPP_TRANSFORMATIONS_PACKING_MFFD_H_
