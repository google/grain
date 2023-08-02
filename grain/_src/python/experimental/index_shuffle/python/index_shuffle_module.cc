#include <pybind11/pybind11.h>

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(index_shuffle_module, m) {
  constexpr char kDoc[] =
      "Returns the position of `index` in a permutation of [0, ..., "
      "max_index].";
  m.doc() = kDoc;
  m.def("index_shuffle", &::grain::random::index_shuffle, kDoc,
        py::arg("index"), py::arg("max_index"), py::arg("seed"),
        py::arg("rounds"));
}
