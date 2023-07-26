load("//tools/build_defs/license:license.bzl", "license")
load("//devtools/deps/check:deps_check.bzl", "check_dependencies")

package(default_visibility = ["//grain:__subpackages__"])

license(
    name = "license",
    package_name = "grain",
)

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//third_party/py/grain/...",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:random",
        "//grain/_src/core:sharding",
    ],
)

check_dependencies(
    name = "core_deps_test",
    of = [
        ":core",
    ],
    dont_depend_on = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_library(
    name = "python",
    srcs = ["python.py"],
    srcs_version = "PY3",
    # Implicit build flag
    visibility = ["//visibility:public"],
    deps = [
        ":core",  # build_cleaner: keep
        ":python_experimental",  # build_cleaner: keep
        "//grain/_src/python:checkpoint_handlers",  # build_cleaner: keep
        "//grain/_src/python:data_loader",  # build_cleaner: keep
        "//grain/_src/python:data_sources",  # build_cleaner: keep
        "//grain/_src/python:grain_pool",  # build_cleaner: keep
        "//grain/_src/python:load",  # build_cleaner: keep
        "//grain/_src/python:operations",  # build_cleaner: keep
        "//grain/_src/python:options",  # build_cleaner: keep
        "//grain/_src/python:samplers",  # build_cleaner: keep
    ],
)

check_dependencies(
    name = "python_deps_test",
    of = [
        ":python",
    ],
    dont_depend_on = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_library(
    name = "python_experimental",
    srcs = ["python_experimental.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        ":python_lazy_dataset",  # build_cleaner: keep
        "//grain/_src/python/experimental/example_packing:packing",  # build_cleaner: keep
    ],
)

py_library(
    name = "python_lazy_dataset",
    srcs = ["python_lazy_dataset.py"],
    srcs_version = "PY3",
    # Implicit build flag
    deps = [
        "//grain/_src/python/lazy_dataset",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset:data_sources",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset:grain_pool_v2",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:batch",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:filter",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:flatmap",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:map",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:mix",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:ragged_batch",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:shuffle",  # build_cleaner: keep
        "//grain/_src/python/lazy_dataset/transformations:slice",  # build_cleaner: keep
    ],
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    srcs_version = "PY3",
    # Implicit build flag
    visibility = ["//visibility:public"],
    deps = [
        ":core",
        ":tensorflow_experimental",  # build_cleaner: keep
        "//grain/_src/tensorflow:batching",
        "//grain/_src/tensorflow:checkpoint_handlers",
        "//grain/_src/tensorflow:data_iterators",
        "//grain/_src/tensorflow:data_loaders",
        "//grain/_src/tensorflow:data_sources",
        "//grain/_src/tensorflow:index_dataset",
    ],
)

py_library(
    name = "tensorflow_experimental",
    srcs = ["tensorflow_experimental.py"],
    srcs_version = "PY3",
)
