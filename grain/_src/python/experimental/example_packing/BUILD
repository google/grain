# Experimental transformation for example packing in PyGrain.

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "packing",
    srcs = ["packing.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "//grain/_src/python:record",
        "@abseil-py//absl/logging",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "packing_test",
    srcs = ["packing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":packing",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python:record",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)
