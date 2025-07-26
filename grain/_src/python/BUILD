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
        "@pypi//etils:pkg",
        "@abseil-py//absl/logging",
        "//grain/_src/core:usage_logging",
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
    name = "multiprocessing_common",
    srcs = [
        "multiprocessing_common.py",
    ],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/dataset:stats"],
)

py_test(
    name = "multiprocessing_common_test",
    srcs = [
        "multiprocessing_common_test.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":multiprocessing_common",
        "@abseil-py//absl/testing:absltest",
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
        ":shared_memory_array",
        "//grain/_src/core:tree_lib",
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
        "//grain/_src/python/dataset:stats",
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
        ":checkpointing",
        ":data_sources",
        ":grain_pool",
        ":operations",
        ":options",
        ":record",
        ":samplers",
        ":shared_memory_array",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "//grain/_src/core:tree_lib",
        "//grain/_src/core:usage_logging",
        "@abseil-py//absl/logging",
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
        ":shared_memory_array",
        "//grain/_src/core:sharding",
        "//grain/_src/core:transforms",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
        "@pypi//parameterized:pkg",
    ],
)

py_library(
    name = "grain_pool",
    srcs = ["grain_pool.py"],
    srcs_version = "PY3",
    deps = [
        ":grain_logging",
        ":multiprocessing_common",
        ":options",
        ":record",
        ":shared_memory_array",
        "//grain/_src/core:config",
        "//grain/_src/core:monitoring",
        "//grain/_src/core:parallel",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset:stats",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/logging",
        "@pypi//cloudpickle:pkg",
    ],
)

py_test(
    name = "grain_pool_test",
    srcs = ["grain_pool_test.py"],
    shard_count = 20,
    srcs_version = "PY3",
    tags = ["not_run:arm"],
    deps = [
        ":data_sources",
        ":grain_pool",
        ":options",
        ":record",
        "//grain/_src/core:config",
        "//grain/_src/core:monitoring",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_library(
    name = "checkpoint_handlers",
    srcs = ["checkpoint_handlers.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        "//grain/_src/core:sharding",
        "//grain/_src/python/dataset",
        "@pypi//etils:pkg",
    ],
)

py_library(
    name = "load",
    srcs = ["load.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loader",
        ":data_sources",
        ":options",
        ":samplers",
        "//grain/_src/core:monitoring",
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
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "shared_memory_array",
    srcs = ["shared_memory_array.py"],
    srcs_version = "PY3",
    deps = [
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "shared_memory_array_test",
    srcs = ["shared_memory_array_test.py"],
    srcs_version = "PY3",
    deps = [
        ":operations",
        ":record",
        ":shared_memory_array",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//jax:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "checkpointing",
    srcs = ["checkpointing.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "@pypi//etils:pkg",
    ],
)
