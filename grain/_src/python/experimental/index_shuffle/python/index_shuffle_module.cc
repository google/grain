// #include <pybind11/pybind11.h>

// #include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

// namespace py = pybind11;

// PYBIND11_MODULE(index_shuffle_module, m) {
//   constexpr char kDoc[] =
//       "Returns the position of `index` in a permutation of [0, ..., "
//       "max_index].";
//   m.doc() = kDoc;
//   m.def("index_shuffle", &::grain::random::index_shuffle, kDoc,
//         py::arg("index"), py::arg("max_index"), py::arg("seed"),
//         py::arg("rounds"));
// }

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>  // Include this if you use std::function in bindings

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(index_shuffle_module, m) {
  constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  m.doc() = kDoc;
  m.def("index_shuffle", [](int64_t index, int64_t max_index, int64_t seed, int64_t rounds) {
    py::gil_scoped_release release;  // Release the GIL
    auto result = ::grain::random::index_shuffle(index, max_index, seed, rounds);
    py::gil_scoped_acquire acquire;  // Re-acquire the GIL before returning to Python
    return result;
  }, kDoc, py::arg("index"), py::arg("max_index"), py::arg("seed"), py::arg("rounds"));
}