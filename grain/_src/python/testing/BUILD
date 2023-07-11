# Package for Grain mocking library.

pytype_strict_library(
    name = "mocking",
    srcs = ["mocking.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//file/google_src",
        "//file/localfile",
        "//grain/_src/python:data_sources",
    ],
)

pytype_strict_test(
    name = "mocking_test",
    timeout = "long",
    srcs = ["mocking_test.py"],
    data = ["//third_party/py/tensorflow_datasets/testing/metadata:wmt17_translate"],
    deps = [
        ":mocking",
        "//grain:python",
    ],
)
