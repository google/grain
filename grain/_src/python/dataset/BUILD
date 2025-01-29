package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "base",
    srcs = [
        "base.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//grain/_src/python:data_sources",
    ],
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
        "//grain/_src/python/dataset/transformations:core_transformations",
    ],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":stats",
        "//grain/_src/core:exceptions",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:transforms",
        "//grain/_src/core:tree_lib",
        "//grain/_src/core:usage_logging",
        "//grain/_src/python:grain_pool",
        "//grain/_src/python:options",
        "//grain/_src/python:shared_memory_array",
    ],
)

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":dataset",
        ":stats",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
    ],
)

py_library(
    name = "visualize",
    srcs = ["visualize.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python:options",
    ],
)

py_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":visualize",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//grain/_src/core:config",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:tree_lib",
    ],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":stats",
        "//grain/_src/core:transforms",
    ],
)
