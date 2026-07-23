#include <pybind11/pybind11.h>

#include <cstdint>
#include <string>

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(index_shuffle_module, m) {
  static constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  m.doc() = kDoc;
  m.def(
      "index_shuffle",
      [](int64_t index, int64_t max_index, uint32_t seed, uint32_t rounds) {
        if (rounds < 4 || rounds % 2 != 0 || rounds > 1024) {
          // Using std::to_string to avoid extra dependency in setup.py.
          throw py::value_error(
              "rounds must be an even integer between 4 and 1024, "
              "but got rounds = " +
              std::to_string(rounds));
        }
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
}
