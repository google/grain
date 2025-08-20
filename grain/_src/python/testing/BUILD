load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "experimental",
    srcs = ["experimental.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "@pypi//numpy:pkg",
    ],
)
