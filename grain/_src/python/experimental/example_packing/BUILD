# Experimental transformation for example packing in PyGrain.
load("@pypi_grain//:requirements.bzl", "requirement")

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "packing",
    srcs = ["packing.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree",
        "//grain/_src/python:record",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name = "packing_test",
    srcs = ["packing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":packing",
        "//grain/_src/core:tree",
        "//grain/_src/python:record",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow"), requirement("jaxtyping"), 
    ],
)
