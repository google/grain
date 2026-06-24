package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "@pypi//etils:pkg",
    ],
)

py_library(
    name = "handler",
    srcs = ["handler.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "//grain/_src/python:data_loader",
        "//grain/_src/python/dataset",
        "@pypi//etils:pkg",
    ],
)
