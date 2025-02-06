load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "experimental",
    srcs = ["experimental.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "@pypi//numpy:pkg",
    ],
)
