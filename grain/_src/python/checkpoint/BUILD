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
        ":elastic_checkpoint",
        "//grain/_src/core:sharding",
        "//grain/_src/python:data_loader",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:elastic_iterator",
        "@pypi//etils:pkg",
    ],
)

py_library(
    name = "elastic_checkpoint",
    srcs = ["elastic_checkpoint.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset:elastic_iterator",
        "@pypi//etils:pkg",
    ],
)

py_test(
    name = "elastic_checkpoint_test",
    srcs = ["elastic_checkpoint_test.py"],
    srcs_version = "PY3",
    deps = [
        ":elastic_checkpoint",
        "//grain/_src/core:sharding",
        "//grain/_src/python/dataset:elastic_iterator",
        "@abseil-py//absl/testing:absltest",
        "@pypi//etils:pkg",
    ],
)
