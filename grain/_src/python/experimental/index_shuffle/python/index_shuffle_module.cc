#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

namespace {

void ValidateRounds(uint32_t rounds) {
  if (rounds < 4 || rounds % 2 != 0 || rounds > 1024) {
    // Using std::to_string to avoid extra dependency in setup.py.
    throw py::value_error(
        "rounds must be an even integer between 4 and 1024, "
        "but got rounds = " +
        std::to_string(rounds));
  }
}

}  // namespace

PYBIND11_MODULE(index_shuffle_module, m) {
  static constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  static constexpr char kRangeDoc[] =
      "Returns the positions of indices in `[start, stop)` in a permutation of "
      "[0, ..., max_index].";
  m.doc() = kDoc;
  m.def(
      "index_shuffle",
      [](int64_t index, int64_t max_index, uint32_t seed, uint32_t rounds) {
        ValidateRounds(rounds);
        if (index < 0 || index > max_index) {
          // Using std::to_string to avoid extra dependency in setup.py.
          throw py::value_error(
              "index must be in [0, max_index], but got index = " +
              std::to_string(index) +
              " and max_index = " + std::to_string(max_index));
        }
        return grain::random::index_shuffle(index, max_index, seed, rounds);
      },
      kDoc, py::arg("index"), py::arg("max_index"), py::arg("seed"),
      py::arg("rounds"), py::call_guard<py::gil_scoped_release>());
  m.def(
      "index_shuffle_range",
      [](int64_t start, int64_t stop, int64_t max_index, uint32_t seed,
         uint32_t rounds) -> std::vector<uint64_t> {
        ValidateRounds(rounds);
        if (max_index < 0) {
          throw py::value_error(
              "max_index must be non-negative, but got max_index = " +
              std::to_string(max_index));
        }
        if (start < 0 || stop < 0) {
          throw py::value_error(
              "start and stop must be non-negative, but got start = " +
              std::to_string(start) + " and stop = " + std::to_string(stop));
        }
        if (start >= stop) {
          return {};
        }
        if (stop - 1 > max_index) {
          throw py::value_error(
              "[start, stop) must be within [0, max_index + 1], but got "
              "start = " +
              std::to_string(start) + ", stop = " + std::to_string(stop) +
              ", and max_index = " + std::to_string(max_index));
        }
        const uint64_t count =
            static_cast<uint64_t>(stop) - static_cast<uint64_t>(start);
        if (count > std::vector<uint64_t>().max_size() ||
            count > static_cast<uint64_t>(PY_SSIZE_T_MAX)) {
          throw std::bad_alloc();
        }
        return grain::random::index_shuffle_range(
            static_cast<uint64_t>(start), static_cast<uint64_t>(stop),
            static_cast<uint64_t>(max_index), seed, rounds);
      },
      kRangeDoc, py::arg("start"), py::arg("stop"), py::arg("max_index"),
      py::arg("seed"), py::arg("rounds"),
      py::call_guard<py::gil_scoped_release>());
}
