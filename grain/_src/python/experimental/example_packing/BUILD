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
    name = "pack_and_batch_test",
    srcs = ["pack_and_batch_test.py"],
    srcs_version = "PY3",
    # Skip bazel test due to JAX installation issue. This is tested with pytest instead.
    target_compatible_with = select({
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":packing",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python:record",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)
