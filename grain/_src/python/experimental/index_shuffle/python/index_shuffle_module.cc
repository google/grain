#include <pybind11/pybind11.h>

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(index_shuffle_module, m) {
  constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  m.doc() = kDoc;
  m.def(
      "index_shuffle",
      [](int64_t index, int64_t max_index, uint32_t seed, uint32_t rounds) {
        if (rounds < 4 || rounds % 2 != 0) {
          throw py::value_error(absl::StrCat(
              "rounds must be an even integer >= 4, but got rounds = ",
              rounds));
        }
        if (index < 0 || index > max_index) {
          throw py::value_error(absl::StrCat(
              "index must be in [0, max_index], but got index = ", index,
              " and max_index = ", max_index));
        }
        return grain::random::index_shuffle(index, max_index, seed, rounds);
      },
      kDoc, py::arg("index"), py::arg("max_index"), py::arg("seed"),
      py::arg("rounds"));
}
