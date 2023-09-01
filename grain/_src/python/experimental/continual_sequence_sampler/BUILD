package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

py_library(
    name = "continual_sequence_sampler",
    srcs = ["continual_sequence_sampler.py"],
    srcs_version = "PY3",
    deps = ["//grain/_src/python:record"],
)

py_test(
    name = "continual_sequence_sampler_test",
    srcs = ["continual_sequence_sampler_test.py"],
    srcs_version = "PY3",
    deps = [":continual_sequence_sampler"],
)
