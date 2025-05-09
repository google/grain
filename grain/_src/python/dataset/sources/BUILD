package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "parquet_dataset",
    srcs = ["parquet_dataset.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/python/dataset",
        "@pypi//etils:pkg",
    ],
)

py_test(
    name = "parquet_dataset_test",
    srcs = ["parquet_dataset_test.py"],
    srcs_version = "PY3",
    deps = [
        ":parquet_dataset",
        "//grain:python",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
        "@pypi//pyarrow:pkg",
    ],
)

py_library(
    name = "tfrecord_dataset",
    srcs = ["tfrecord_dataset.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python/dataset"],
)

py_test(
    name = "tfrecord_dataset_test",
    srcs = ["tfrecord_dataset_test.py"],
    args = ["--test_srcdir=grain/_src/python/testdata"],
    data = [
        "//grain/_src/python/testdata:morris_sequence_first_5.tfrecord",
    ],
    srcs_version = "PY3",
    deps = [
        ":tfrecord_dataset",
        "//grain",
        "@abseil-py//absl/flags",
        "@abseil-py//absl/testing:absltest",
    ],
)
