package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "queue",
    srcs = [
        "queue.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "queue_test",
    srcs = [
        "queue_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":queue",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "shared_memory_array",
    srcs = ["shared_memory_array.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "shared_memory_array_test",
    srcs = ["shared_memory_array_test.py"],
    srcs_version = "PY3",
    # Skip bazel test due to JAX installation issue. This is tested with pytest instead.
    target_compatible_with = select({
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":shared_memory_array",
        "//grain/_src/python:operations",
        "//grain/_src/python:record",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//jax:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "variable_size_queue",
    srcs = ["variable_size_queue.py"],
    srcs_version = "PY3",
)

py_test(
    name = "variable_size_queue_test",
    srcs = ["variable_size_queue_test.py"],
    srcs_version = "PY3",
    deps = [
        ":variable_size_queue",
        "@abseil-py//absl/testing:absltest",
    ],
)
