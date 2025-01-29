package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
    deps = [
        ":monitoring",
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
    deps = [":parallel"],
)

py_library(
    name = "grain_random",
    srcs = ["grain_random.py"],
    srcs_version = "PY3",
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

py_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    srcs_version = "PY3",
    deps = [":transforms"],
)

py_library(
    name = "tree_lib",
    srcs = [
        "tree_lib.py",
    ],
    srcs_version = "PY3",
)

py_library(
    name = "tree_test_lib",
    testonly = 1,
    srcs = ["tree_lib_test.py"],
    srcs_version = "PY3",
    deps = [":tree_lib"],
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
    deps = [
        ":tree_lib",
        ":tree_test_lib",
    ],
)
