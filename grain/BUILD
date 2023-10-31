
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:grain_random",
        "//grain/_src/core:sharding",
    ],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    srcs_version = "PY3",
    # Implicit build flag
    visibility = ["//visibility:public"],
    deps = [
        ":core",  # build_cleaner: keep
        ":python_experimental",  # build_cleaner: keep
        "//grain/_src/core:transforms",  # build_cleaner: keep
        "//grain/_src/python:checkpoint_handlers",  # build_cleaner: keep
        "//grain/_src/python:data_loader",  # build_cleaner: keep
        "//grain/_src/python:data_sources",  # build_cleaner: keep
        "//grain/_src/python:grain_pool",  # build_cleaner: keep
        "//grain/_src/python:load",  # build_cleaner: keep
        "//grain/_src/python:operations",  # build_cleaner: keep
        "//grain/_src/python:options",  # build_cleaner: keep
        "//grain/_src/python:samplers",  # build_cleaner: keep
    ],
)

py_library(
    name = "python_experimental",
    srcs = ["python_experimental.py"],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        ":python_lazy_dataset",  # build_cleaner: keep
        "//grain/_src/core:transforms",  # build_cleaner: keep
        "//grain/_src/python/experimental/example_packing:packing",  # build_cleaner: keep
        "//grain/_src/python/experimental/proto_parsers:encode",  # build_cleaner: keep
        "//grain/_src/python/experimental/proto_parsers:fast_proto_parser",  # build_cleaner: keep
    ],
)

py_library(
    name = "python_lazy_dataset",
    srcs = ["python_lazy_dataset.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/python/lazy_dataset",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset:data_sources",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset:grain_pool_v2",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:batch",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:filter",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:flatmap",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:map",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:mix",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:packing",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:ragged_batch",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:repeat",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:shuffle",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:slice",  # build_cleaner: keep
    ],
)
