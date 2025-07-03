load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@bazel_skylib//lib:selects.bzl", "selects")

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
    data = select({
        "@platforms//os:windows": [":index_shuffle_module.pyd"],
        "//conditions:default": [":index_shuffle_module.so"],
    }),
    srcs_version = "PY3",
    deps = ["@abseil-py//absl/testing:absltest"],
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
)
