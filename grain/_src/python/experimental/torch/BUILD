package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "batch",
    srcs = ["batch.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset:base",
    ],
)

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

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python:options",
    ],
)

py_library(
    name = "prefetch",
    srcs = ["prefetch.py"],
    srcs_version = "PY3",
    deps = [
        ":options",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/dataset:stats",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
    deps = [
        ":batch",
        ":checkpoint",
        ":prefetch",
        ":random_map",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/checkpoint:base",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@pypi//etils:pkg",
        "@pypi//numpy:pkg",
    ],
)
