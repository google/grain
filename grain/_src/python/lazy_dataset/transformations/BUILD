package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

# Transformations that are considered core and are supported by the convenient API in
# LazyMapDataset or LazyIterDataset.
filegroup(
    name = "core_transformations",
    srcs = [
        "filter.py",
        "slice.py",
    ],
)

py_library(
    name = "batch",
    srcs = ["batch.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "batch_test",
    srcs = ["batch_test.py"],
    srcs_version = "PY3",
    deps = [
        ":batch",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "flatmap",
    srcs = ["flatmap.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "flatmap_test",
    srcs = ["flatmap_test.py"],
    srcs_version = "PY3",
    deps = [
        ":flatmap",
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "map",
    srcs = ["map.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "map_test",
    srcs = ["map_test.py"],
    srcs_version = "PY3",
    deps = [
        ":map",
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "mix",
    srcs = ["mix.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:exceptions",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "mix_test",
    srcs = ["mix_test.py"],
    srcs_version = "PY3",
    deps = [
        ":mix",
        ":repeat",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "packing",
    srcs = ["packing.py"],
    srcs_version = "PY3",
    deps = [
        ":packing_packed_batch",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "packing_test",
    srcs = ["packing_test.py"],
    srcs_version = "PY3",
    deps = [
        ":map",
        ":packing",
        ":repeat",
        ":shuffle",
        "//grain/_src/python/lazy_dataset",
        "//grain/_src/python/lazy_dataset:data_sources",
    ],
)

py_library(
    name = "packing_packed_batch",
    srcs = ["packing_packed_batch.py"],
    srcs_version = "PY3",
)

py_test(
    name = "slice_test",
    srcs = ["slice_test.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/lazy_dataset"],
)

py_library(
    name = "shuffle",
    srcs = ["shuffle.py"],
    data = ["//grain/_src/python/experimental/index_shuffle/python:index_shuffle_module.so"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/lazy_dataset"],
)

py_test(
    name = "shuffle_test",
    srcs = ["shuffle_test.py"],
    srcs_version = "PY3",
    deps = [
        ":shuffle",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "ragged_batch",
    srcs = ["ragged_batch.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_test(
    name = "ragged_batch_test",
    srcs = ["ragged_batch_test.py"],
    srcs_version = "PY3",
    deps = [
        ":map",
        ":ragged_batch",
        "//grain/_src/core:transforms",
        "//grain/_src/python/lazy_dataset",
        "//grain/_src/python/lazy_dataset:data_sources",
    ],
)

py_library(
    name = "repeat",
    srcs = ["repeat.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/lazy_dataset"],
)

py_test(
    name = "repeat_test",
    srcs = ["repeat_test.py"],
    srcs_version = "PY3",
    deps = [
        ":repeat",
        "//grain/_src/python/lazy_dataset",
    ],
)

py_library(
    name = "zip",
    srcs = ["zip.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/lazy_dataset"],
)

py_test(
    name = "zip_test",
    srcs = ["zip_test.py"],
    srcs_version = "PY3",
    deps = [
        ":zip",
        "//grain/_src/python/lazy_dataset",
    ],
)
