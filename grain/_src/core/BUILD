package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
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
    deps = [":parallel"],
)

py_library(
    name = "grain_random",
    srcs = ["grain_random.py"],
    srcs_version = "PY3",
)

py_test(
    name = "grain_random_test",
    srcs = ["grain_random_test.py"],
    srcs_version = "PY3",
    deps = [":grain_random"],
)

py_library(
    name = "sharding",
    srcs = ["sharding.py"],
    srcs_version = "PY3",
)

py_test(
    name = "sharding_test",
    srcs = ["sharding_test.py"],
    srcs_version = "PY3",
    deps = [":sharding"],
)

py_library(
    name = "usage_logging",
    srcs = ["usage_logging.py"],
    srcs_version = "PY3",
)

py_library(
    name = "transforms",
    srcs = ["transforms.py"],
    srcs_version = "PY3",
)
