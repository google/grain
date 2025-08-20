load("@rules_python//python:defs.bzl", "py_library", "py_test")

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

pybind_extension(
    name = "index_shuffle_module",
    srcs = ["index_shuffle_module.cc"],
    deps = [
        "//grain/_src/python/experimental/index_shuffle",
    ],
)

py_test(
    name = "index_shuffle_test",
    srcs = ["index_shuffle_test.py"],
    srcs_version = "PY3",
    data = select({
        "@platforms//os:windows": [":index_shuffle_module.pyd"],
        "//conditions:default": [":index_shuffle_module.so"],
    }),
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":index_shuffle_module",
        "@abseil-py//absl/testing:absltest",
    ],
)
