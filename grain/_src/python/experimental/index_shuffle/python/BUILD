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
    data = [":index_shuffle_module.so"],
    srcs_version = "PY3",
    deps = ["@abseil-py//absl/testing:absltest"],
)
