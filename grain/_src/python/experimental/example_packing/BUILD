# Experimental transformation for example packing in PyGrain.

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "packing",
    srcs = ["packing.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python:record",
        "//third_party/py/jaxtyping",
    ],
)

py_test(
    name = "packing_test",
    srcs = ["packing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":packing",
        "//grain/_src/python:record",
    ],
)
