package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "lazy_dataset",
    srcs = ["lazy_dataset.py"],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "//grain/_src/python:options",
    ],
)

py_test(
    name = "lazy_dataset_test",
    srcs = ["lazy_dataset_test.py"],
    srcs_version = "PY3",
    deps = [
        ":lazy_dataset",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/lazy_dataset/transformations:filter",
    ],
)

py_library(
    name = "data_sources",
    srcs = ["data_sources.py"],
    srcs_version = "PY3",
    deps = [
        ":lazy_dataset",
        "//grain/_src/python:options",
        "//grain/google/lineage:lineage_logging",
    ],
)

py_test(
    name = "data_sources_test",
    srcs = ["data_sources_test.py"],
    srcs_version = "PY3",
    deps = [":data_sources"],
)

py_library(
    name = "grain_pool_v2",
    srcs = ["grain_pool_v2.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:parallel",
        "//grain/_src/python:multiprocessing_common",
    ],
)

py_test(
    name = "grain_pool_v2_test",
    srcs = ["grain_pool_v2_test.py"],
    srcs_version = "PY3",
    deps = [
        ":grain_pool_v2",
        "//grain/_src/core:sharding",
        "//grain/_src/python:samplers",
    ],
)