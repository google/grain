package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "continual_sequence_sampler",
    srcs = ["continual_sequence_sampler.py"],
    srcs_version = "PY3",
    deps = [
        "//grain/_src/core:sharding",
        "//grain/_src/python:record",
        "//grain/_src/python/lazy_dataset",
        "//grain/_src/python/lazy_dataset/transformations:shuffle",
    ],
)

py_test(
    name = "continual_sequence_sampler_test",
    srcs = ["continual_sequence_sampler_test.py"],
    srcs_version = "PY3",
    deps = [
        ":continual_sequence_sampler",
        "//grain/_src/core:sharding",
        "//grain/_src/python/lazy_dataset",
        "//grain/_src/python/lazy_dataset/transformations:shuffle",
    ],
)
