load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pypi_grain//:requirements.bzl", "requirement")
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
    data = [
        ":index_shuffle_module.so",
    ],
    deps = [
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
    srcs_version = "PY3"
)
