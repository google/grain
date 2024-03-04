# Off the shelf LazyDataset data sources.

package(default_visibility = ["//grain:__subpackages__"])

py_library(
    name = "tfgrain",
    srcs = ["tfgrain.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//grain:tensorflow",
        "//grain/_src/python/lazy_dataset",
        "//third_party/py/clu/data:dataset_iterator",
    ],
)

py_test(
    name = "tfgrain_test",
    srcs = ["tfgrain_test.py"],
    srcs_version = "PY3",
    deps = [
        ":tfgrain",
        "//grain:tensorflow",
    ],
)
