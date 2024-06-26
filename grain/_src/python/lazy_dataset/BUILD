package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "lazy_dataset",
    srcs = [
        "lazy_dataset.py",
        "//grain/_src/python/lazy_dataset/transformations:core_transformations",
    ],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:tree",
        "//grain/_src/core:usage_logging",
        "//grain/_src/python:grain_pool",
        "//grain/_src/python:options",
        "//grain/_src/python:shared_memory_array",
    ],
)

py_test(
    name = "lazy_dataset_test",
    srcs = ["lazy_dataset_test.py"],
    shard_count = 20,
    srcs_version = "PY3",
    deps = [
        ":lazy_dataset",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/lazy_dataset/transformations:map",
    ],
)

py_library(
    name = "data_sources",
    srcs = ["data_sources.py"],
    srcs_version = "PY3",
    deps = [
        ":lazy_dataset",
    ],
)

py_test(
    name = "data_sources_test",
    srcs = ["data_sources_test.py"],
    srcs_version = "PY3",
    deps = [
        ":data_sources",
        ":lazy_dataset",
    ],
)

py_library(
    name = "data_loader",
    srcs = ["data_loader.py"],
    srcs_version = "PY3",
    deps = [
        ":lazy_dataset",
        "//grain/_src/python:options",
    ],
)

py_test(
    name = "data_loader_test",
    srcs = ["data_loader_test.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        "//grain:python_experimental",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
    ],
)
