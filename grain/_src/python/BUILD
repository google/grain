package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "data_sources",
    srcs = [
        "data_sources.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//file/base:options_py_pb2",
        "//grain/_src/core:usage_logging",
    ],
)

py_test(
    name = "data_sources_test",
    srcs = [
        "data_sources_test.py",
    ],
    args = ["--test_srcdir=grain/_src/python"],
    data = [
        "//grain/_src/python/testdata:digits.array_record-00000-of-00002",
        "//grain/_src/python/testdata:digits.array_record-00001-of-00002",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_sources",
    ],
)

py_library(
    name = "record",
    srcs = [
        "record.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "record_test",
    srcs = [
        "record_test.py",
    ],
    srcs_version = "PY3",
    deps = [":record"],
)

py_library(
    name = "multiprocessing_common",
    srcs = [
        "multiprocessing_common.py",
    ],
    srcs_version = "PY3",
)

py_test(
    name = "multiprocessing_common_test",
    srcs = [
        "multiprocessing_common_test.py",
    ],
    srcs_version = "PY3",
    deps = [":multiprocessing_common"],
)

py_library(
    name = "operations",
    srcs = [
        "operations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":record",
        ":shared_memory_array",
    ],
)

py_test(
    name = "operations_test",
    srcs = [
        "operations_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":record",
    ],
)

py_library(
    name = "samplers",
    srcs = [
        "samplers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":record",
        "//grain/_src/core:sharding",
        "//grain/_src/python/lazy_dataset",
        "//grain/_src/python/lazy_dataset/transformations:shuffle",
    ],
)

py_test(
    name = "samplers_test",
    srcs = [
        "samplers_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":record",
        ":samplers",
        "//grain/_src/core:sharding",
    ],
)

py_library(
    name = "data_loader",
    srcs = [
        "data_loader.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_sources",
        ":grain_pool",
        ":multiprocessing_common",
        ":np_array_in_shared_memory",
        ":operations",
        ":options",
        ":record",
        ":samplers",
        ":shared_memory_array",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:usage_logging",
    ],
)

py_test(
    name = "data_loader_test",
    srcs = [
        "data_loader_test.py",
    ],
    args = ["--test_srcdir=grain/_src/python"],
    data = [
        "//grain/_src/python/testdata:digits.array_record-00000-of-00002",
        "//grain/_src/python/testdata:digits.array_record-00001-of-00002",
    ],
    shard_count = 20,
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        ":data_sources",
        ":operations",
        ":samplers",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
    ],
)

py_library(
    name = "grain_pool",
    srcs = ["grain_pool.py"],
    srcs_version = "PY3",
    deps = [
        ":grain_logging",
        ":multiprocessing_common",
        ":np_array_in_shared_memory",
        ":options",
        "//grain/_src/core:parallel",
    ],
)

py_test(
    name = "grain_pool_test",
    srcs = ["grain_pool_test.py"],
    srcs_version = "PY3",
    tags = ["not_run:arm"],
    deps = [
        ":data_sources",
        ":grain_pool",
        ":options",
    ],
)

py_library(
    name = "checkpoint_handlers",
    srcs = ["checkpoint_handlers.py"],
    srcs_version = "PY3",
    deps = [":data_loader"],
)

py_library(
    name = "load",
    srcs = ["load.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        ":data_sources",
        ":operations",
        ":options",
        ":samplers",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:usage_logging",
    ],
)

py_test(
    name = "load_test",
    srcs = ["load_test.py"],
    srcs_version = "PY3",
    deps = [
        ":data_sources",
        ":load",
        "//grain/_src/core:transforms",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
)

py_library(
    name = "grain_logging",
    srcs = ["grain_logging.py"],
    srcs_version = "PY3",
)

py_test(
    name = "grain_logging_test",
    srcs = ["grain_logging_test.py"],
    srcs_version = "PY3",
    deps = [":grain_logging"],
)

py_library(
    name = "np_array_in_shared_memory",
    srcs = ["np_array_in_shared_memory.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python:shared_memory_array"],
)

py_test(
    name = "np_array_in_shared_memory_test",
    srcs = ["np_array_in_shared_memory_test.py"],
    srcs_version = "PY3",
    deps = [
        ":np_array_in_shared_memory",
        ":shared_memory_array",
    ],
)

py_library(
    name = "shared_memory_array",
    srcs = ["shared_memory_array.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shared_memory_array_test",
    srcs = ["shared_memory_array_test.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":record",
        ":shared_memory_array",
    ],
)
