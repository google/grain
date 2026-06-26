
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

# Backwards compatibility alias.
alias(
    name = "python",
    actual = ":grain",
    visibility = ["//visibility:public"],
)

alias(
    name = "prism",
    actual = "//third_party/py/grain/_src/prism:debug_server",
    visibility = [
        "//learning/gemini/cms:users",
        "//learning/gemini/gemax:users",
        "//learning/gemini/prod:users",
    ],
)

py_library(
    name = "grain",
    srcs = [
        "__init__.py",
        "_src/__init__.py",
        "checkpoint.py",
        "constants.py",
        "experimental.py",
        "multiprocessing.py",
        "python/__init__.py",
        "python/experimental.py",
        "samplers.py",
        "sharding.py",
        "sources.py",
        "transforms.py",
    ],
    srcs_version = "PY3",
    data = select({
        "@platforms//os:windows": ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.pyd"],
       "//conditions:default": ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    }),
    # Implicit build flag
    visibility = ["//visibility:public"],
    deps = [
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:version",
        "//grain/_src/python/checkpoint:handler",
        "//grain/_src/python:data_loader",
        "//grain/_src/python:data_sources",
        "//grain/_src/python:load",
        "//grain/_src/python:operations",
        "//grain/_src/python:options",
        "//grain/_src/python:record",
        "//grain/_src/python:samplers",
        "//grain/_src/python/ipc:shared_memory_array",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/dataset:elastic_iterator",
        "//grain/_src/python/dataset:stats",
        "//grain/_src/python/dataset/sources:parquet_dataset",
        "//grain/_src/python/dataset/sources:tfrecord_dataset",
        "//grain/_src/python/dataset/transformations:cache",
        "//grain/_src/python/dataset/transformations:prefetch_autotune",
        "//grain/_src/python/dataset/transformations:limit",
        "//grain/_src/python/dataset/transformations:packing",
        "//grain/_src/python/dataset/transformations:packing_concat_then_split",
        "//grain/_src/python/dataset/transformations:zip",
        "//grain/_src/python/experimental/device_put",
        "//grain/_src/python/experimental/example_packing:packing",
        "//grain/_src/python/testing:experimental",
    ],
)
