package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

# Transformations that are considered core and are supported by the convenient API in
# MapDataset or IterDataset.
filegroup(
    name = "core_transformations",
    srcs = [
        "batch.py",
        "filter.py",
        "flatmap.py",
        "interleave.py",
        "map.py",
        "mix.py",
        "prefetch.py",
        "process_prefetch.py",
        "rebatch.py",
        "repeat.py",
        "shuffle.py",
        "slice.py",
        "source.py",
    ],
)

py_test(
    name = "batch_test",
    srcs = ["batch_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//jax:pkg",  # buildcleaner: keep
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "prefetch_test",
    timeout = "long",
    srcs = ["prefetch_test.py"],
    shard_count = 50,
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:config",
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "source_test",
    srcs = ["source_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "testing_util",
    testonly = 1,
    srcs = ["testing_util.py"],
    srcs_version = "PY3",
    deps = [
        ":packing",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//jax:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "prefetch_autotune",
    srcs = ["prefetch_autotune.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "@abseil-py//absl/logging",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "flatmap_test",
    srcs = ["flatmap_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "map_test",
    srcs = ["map_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//cloudpickle:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "mix_test",
    srcs = ["mix_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "packing",
    srcs = ["packing.py"],
    srcs_version = "PY3",
    deps = [
        ":packing_packed_batch",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
        "@pypi//etils:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "packing_test",
    srcs = ["packing_test.py"],
    srcs_version = "PY3",
    # Skip bazel test due to JAX installation issue. This is tested with pytest instead.
    target_compatible_with = select({
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":packing",
        ":testing_util",
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "packing_packed_batch",
    srcs = ["packing_packed_batch.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "slice_test",
    srcs = ["slice_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
        "@pypi//typing_extensions:pkg",
    ],
)

py_test(
    name = "shuffle_test",
    srcs = ["shuffle_test.py"],
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "repeat_test",
    srcs = ["repeat_test.py"],
    shard_count = 5,
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
        "@pypi//typing_extensions:pkg",
    ],
)

py_library(
    name = "zip",
    srcs = ["zip.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
    ],
)

py_test(
    name = "zip_test",
    srcs = ["zip_test.py"],
    srcs_version = "PY3",
    deps = [
        ":zip",
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "interleave_test",
    srcs = ["interleave_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:flagsaver",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "limit",
    srcs = ["limit.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
    ],
)

py_test(
    name = "limit_test",
    srcs = ["limit_test.py"],
    srcs_version = "PY3",
    deps = [
        ":limit",
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "packing_concat_then_split",
    srcs = ["packing_concat_then_split.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:exceptions",
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
        "@abseil-py//absl/logging",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "packing_concat_then_split_test",
    srcs = ["packing_concat_then_split_test.py"],
    srcs_version = "PY3",
    deps = [
        ":packing_concat_then_split",
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "rebatch_test",
    srcs = ["rebatch_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:tree_lib",
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "cache",
    srcs = ["cache.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
    ],
)

py_test(
    name = "cache_test",
    srcs = ["cache_test.py"],
    srcs_version = "PY3",
    deps = [
        ":cache",
        ":limit",
        "//grain",
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "//grain/_src/python/testing:experimental",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "process_prefetch_test",
    timeout = "long",
    srcs = ["process_prefetch_test.py"],
    shard_count = 50,
    srcs_version = "PY3",
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        "@abseil-py//absl/logging",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//numpy:pkg",
    ],
)
