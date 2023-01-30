load("@org_tensorflow//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        # "Implicit dependency absl/flags",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [
    ],
)

pytype_strict_library(
    name = "random",
    srcs = ["grain_random.py"],
    deps = [
        # "Implicit dependency absl/logging",
        # "Implicit dependency jax",
        # "Implicit dependency numpy",
    ],
)

py_strict_test(
    name = "random_test",
    srcs = ["random_test.py"],
    deps = [
        ":random",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency jax",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "sharding",
    srcs = ["sharding.py"],
    deps = [
        # "Implicit dependency absl/logging",
        # "Implicit dependency jax",
    ],
)

py_strict_test(
    name = "sharding_test",
    srcs = ["sharding_test.py"],
    deps = [
        ":sharding",
        # "Implicit dependency absl/testing:absltest",
        # "Implicit dependency absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "usage_logging",
    srcs = ["usage_logging.py"],
    deps = [
        # "Remove dependency //learning/deepmind/analytics:usage_logging",
    ],
)
