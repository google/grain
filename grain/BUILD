
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "python",
    srcs = [
        "__init__.py",
        "_src/__init__.py",
        "python/__init__.py",
        "python/experimental.py",
        "python/fast_proto.py",
    ],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    # Implicit build flag
    visibility = ["//visibility:public"],
    deps = [
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/python:checkpoint_handlers",
        "//grain/_src/python:data_loader",
        "//grain/_src/python:data_sources",
        "//grain/_src/python:load",
        "//grain/_src/python:operations",
        "//grain/_src/python:options",
        "//grain/_src/python:record",
        "//grain/_src/python:samplers",
        "//grain/_src/python:shared_memory_array",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/dataset:visualize",
        "//grain/_src/python/dataset/transformations:flatmap",
        "//grain/_src/python/dataset/transformations:interleave",
        "//grain/_src/python/dataset/transformations:limit",
        "//grain/_src/python/dataset/transformations:packing",
        "//grain/_src/python/dataset/transformations:zip",
        "//grain/_src/python/experimental/example_packing:packing",
        "//grain/_src/python/testing:experimental",
        "//grain/proto:execution_summary_py_pb2",
    ],
)
