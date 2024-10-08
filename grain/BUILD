
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/core:config",  # build_cleaner: keep
        "//grain/_src/core:constants",  # build_cleaner: keep
        "//grain/_src/core:grain_random",  # build_cleaner: keep
        "//grain/_src/core:sharding",  # build_cleaner: keep
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
        "//grain/_src/core:constants",  # build_cleaner: keep
        "//grain/_src/core:monitoring",  # build_cleaner: keep
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
        "//grain/_src/python/dataset:visualize",  # build_cleaner: keep
        "//grain/_src/python/experimental/example_packing:packing",  # build_cleaner: keep
    ],
)

py_library(
    name = "python_lazy_dataset",
    srcs = ["python_lazy_dataset.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/python/dataset",  # build_cleaner: keep
        "//grain/_src/python/dataset/transformations:flatmap",  # build_cleaner: keep
        "//grain/_src/python/dataset/transformations:packing",  # build_cleaner: keep
        "//grain/_src/python/dataset/transformations:zip",  # build_cleaner: keep
    ],
)
