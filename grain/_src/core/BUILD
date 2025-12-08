load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        ":monitoring",
        "@abseil-py//absl/flags",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
    srcs_version = "PY3",
)

py_library(
    name = "exceptions",
    srcs = ["exceptions.py"],
    srcs_version = "PY3",
)

py_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "parallel",
    srcs = ["parallel.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "parallel_test",
    srcs = ["parallel_test.py"],
    srcs_version = "PY3",
    deps = [
        ":parallel",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_library(
    name = "grain_random",
    srcs = ["grain_random.py"],
    srcs_version = "PY3",
    deps = [
        "@abseil-py//absl/logging",
        "@pypi//jax:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "sharding",
    srcs = ["sharding.py"],
    srcs_version = "PY3",
    deps = [
        "@abseil-py//absl/logging",
    ],
)

py_test(
    name = "sharding_test",
    srcs = ["sharding_test.py"],
    srcs_version = "PY3",
    deps = [
        ":sharding",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_library(
    name = "transforms",
    srcs = ["transforms.py"],
    srcs_version = "PY3",
    deps = [
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    srcs_version = "PY3",
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":transforms",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_library(
    name = "tree_lib",
    srcs = [
        "tree_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        "@pypi//etils:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "tree_test_lib",
    testonly = 1,
    srcs = ["tree_lib_test.py"],
    srcs_version = "PY3",
    deps = [
        ":tree_lib",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
        "@pypi//attrs:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_test(
    name = "tree_lib_test",
    srcs = ["tree_lib_test.py"],
    srcs_version = "PY3",
    deps = [
        ":tree_test_lib",
    ],
)

py_test(
    name = "tree_lib_jax_test",
    srcs = ["tree_lib_jax_test.py"],
    srcs_version = "PY3",
    # Skip bazel test due to JAX installation issue. This is tested with pytest instead.
    target_compatible_with = select({
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":tree_lib",
        ":tree_test_lib",
        "@abseil-py//absl/testing:absltest",
        "@pypi//jax:pkg",
        "@pypi//numpy:pkg",
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    srcs_version = "PY3",
)

py_test(
    name = "py_version_test",
    srcs = ["py_version_test.py"],
    srcs_version = "PY3",
    deps = [
        "//grain:python",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    srcs_version = "PY3",
    deps = ["@abseil-py//absl/logging"],
)

py_test(
    name = "profiler_test_no_framework",
    srcs = ["profiler_test.py"],
    env = {
        "EXPECTED_FRAMEWORK": "NO_FRAMEWORK",
    },
    main = "profiler_test.py",
    srcs_version = "PY3",
    deps = [
        ":profiler",
        "@abseil-py//absl/testing:absltest",
    ],
)

py_test(
    name = "profiler_test_with_jax",
    srcs = ["profiler_test.py"],
    env = {
        "EXPECTED_FRAMEWORK": "jax",
    },
    main = "profiler_test.py",
    srcs_version = "PY3",
    deps = [
        ":profiler",
        "@abseil-py//absl/testing:absltest",
        "@pypi//jax:pkg",  # buildcleaner: keep
    ],
)

py_library(
    name = "pytest",
    srcs = [
        "pytest.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "traceback_util",
    srcs = ["traceback_util.py"],
    srcs_version = "PY3",
    tags = ["ignore_for_dep=third_party.py.IPython"],
    deps = [
        ":config",
    ],
)

py_test(
    name = "traceback_util_test",
    srcs = ["traceback_util_test.py"],
    srcs_version = "PY3",
    deps = [
        ":config",
        ":traceback_util",
        "@abseil-py//absl/testing:absltest",
    ],
)
