#include <pybind11/pybind11.h>

#include <cstdint>

#include "third_party/absl/strings/str_format.h"
#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;

uint64_t index_shuffle(uint64_t index, uint64_t max_index, uint32_t seed,
                       uint32_t rounds) {
  if (index > max_index) {
    throw py::value_error(absl::StrFormat("index must be in [0, %d] but was %d",
                                          max_index, index));
  }
  return ::grain::random::index_shuffle(index, max_index, seed, rounds);
}

PYBIND11_MODULE(index_shuffle_module, m) {
  constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  m.doc() = kDoc;
  m.def("index_shuffle", &index_shuffle, kDoc, py::arg("index"),
        py::arg("max_index"), py::arg("seed"), py::arg("rounds"));
}
