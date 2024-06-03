#include <pybind11/pybind11.h>

#include "grain/_src/python/experimental/index_shuffle/index_shuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(index_shuffle_module, m) {
  constexpr char kDoc[] =
      R"(Returns the position of `index` in a permutation of
      `[0, ..., max_index]`.

      This use a Simon block cipher with block size B.
      B depends on `max_index`: `B = max(16, ceil(log2(max_index))` where 16 is
      the hardcoded minimal block size.

      `seed` is used to seed the block cipher which is applied for `rounds`
      rounds. Each combination (`seed`, `rounds`, `B`) produces a random
      permutation of the numbers `[0, 2^B)`.

      `rounds` must be a positive even integer >= 4. Larger values improve
      'randomness' of permutations for small `max_index` values. The time to
      compute the result scales linearly with the number of rounds. We recommend 8
      rounds for a good trade off.
  )";
  m.doc() = "This module implements index_shuffle via random permutations";
  m.def("index_shuffle", &::grain::random::index_shuffle, kDoc,
        py::arg("index"), py::arg("max_index"), py::arg("seed"),
        py::arg("rounds"));
}
