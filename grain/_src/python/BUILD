package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "data_sources",
    srcs = [
        "data_sources.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:monitoring",
        "@abseil-py//absl/logging",
        "@pypi//etils:pkg",
        "//grain/_src/python/dataset:stats",
    ] + select({
        "@platforms//os:windows": [],
        "//conditions:default": ["@pypi//array_record:pkg"],
    }),
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
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":data_sources",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//etils:pkg",
    ],
)

py_library(
    name = "record",
    srcs = [
        "record.py",
    ],
    srcs_version = "PY3",
    deps = [
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "record_test",
    srcs = [
        "record_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":record",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "operations",
    srcs = [
        "operations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":record",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/ipc:shared_memory_array",
        "@abseil-py//absl/logging",
        "@pypi//numpy:pkg",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
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
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/python/dataset",
        "@pypi//numpy:pkg",
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
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_library(
    name = "data_loader",
    srcs = [
        "data_loader.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":options",
        ":record",
        ":samplers",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/checkpoint:base",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/ipc:shared_memory_array",
        "@pypi//etils:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "data_loader_test",
    size = "large",
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
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":data_loader",
        ":data_sources",
        ":operations",
        ":options",
        ":samplers",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        "//grain/_src/python/ipc:shared_memory_array",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
        "@pypi//parameterized:pkg",
    ],
)

py_library(
    name = "load",
    srcs = ["load.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        ":options",
        ":samplers",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset:base",
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
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
    deps = [
        "@abseil-py//absl/logging",
    ],
)

py_test(
    name = "options_test",
    srcs = ["options_test.py"],
    srcs_version = "PY3",
    deps = [
        ":options",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "grain_logging",
    srcs = ["grain_logging.py"],
    srcs_version = "PY3",
    deps = ["@abseil-py//absl/logging"],
)

py_test(
    name = "grain_logging_test",
    srcs = ["grain_logging_test.py"],
    srcs_version = "PY3",
    deps = [
        ":grain_logging",
        "//grain/_src/core:pytest",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
    ],
)
