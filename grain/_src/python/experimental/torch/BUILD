package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "random_map",
    srcs = ["random_map.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
    ],
)

py_library(
    name = "distributed",
    srcs = ["distributed.py"],
    srcs_version = "PY3",
)

py_library(
    name = "checkpoint",
    srcs = ["checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        ":distributed",
        "//grain/_src/python/checkpoint:base",
        "@pypi//etils:pkg",
    ],
)
