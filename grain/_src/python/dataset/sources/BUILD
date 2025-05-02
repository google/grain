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
