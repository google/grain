package(default_visibility=["//grain:__subpackages__"])
load("@pypi_grain//:requirements.bzl", "requirement")
licenses(["notice"])

# Transformations that are considered core and are supported by the convenient API in
# MapDataset or IterDataset.
filegroup(
    name="core_transformations",
    srcs=[
        "batch.py",
        "filter.py",
        "map.py",
        "mix.py",
        "prefetch.py",
        "repeat.py",
        "shuffle.py",
        "slice.py",
        "source.py",
    ],
)

py_test(
    name="batch_test",
    srcs=["batch_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow"), 
    ],
)

py_test(
    name="filter_test",
    srcs=["filter_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="prefetch_test",
    srcs=["prefetch_test.py"],
    shard_count=20,
    srcs_version="PY3",
    deps=[
        "//grain/_src/core:transforms",
        "//grain/_src/python:options",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="source_test",
    srcs=["source_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_library(
    name="flatmap",
    srcs=["flatmap.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
    ],
)

py_test(
    name="flatmap_test",
    srcs=["flatmap_test.py"],
    srcs_version="PY3",
    deps=[
        ":flatmap",
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="map_test",
    srcs=["map_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/core:transforms",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="mix_test",
    srcs=["mix_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:base",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_library(
    name="packing",
    srcs=["packing.py"],
    srcs_version="PY3",
    deps=[
        ":packing_packed_batch",
        "//grain/_src/python/dataset",
        "//grain/_src/python/dataset:stats",
    ],
)

py_test(
    name="packing_test",
    srcs=["packing_test.py"],
    srcs_version="PY3",
    deps=[
        ":packing",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow"), requirement("jaxtyping"), requirement("dm_tree")
    ],
)

py_library(
    name="packing_packed_batch",
    srcs=["packing_packed_batch.py"],
    srcs_version="PY3",
    deps=[],
)

py_test(
    name="slice_test",
    srcs=["slice_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="shuffle_test",
    srcs=["shuffle_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_test(
    name="repeat_test",
    srcs=["repeat_test.py"],
    srcs_version="PY3",
    deps=[
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)

py_library(
    name="zip",
    srcs=["zip.py"],
    srcs_version="PY3",
    deps=["//grain/_src/python/dataset"],
)

py_test(
    name="zip_test",
    srcs=["zip_test.py"],
    srcs_version="PY3",
    deps=[
        ":zip",
        "//grain/_src/python/dataset",
        requirement("absl_py"), requirement("numpy"), requirement("cloudpickle"), requirement("etils"), requirement("jax"), requirement("typing_extensions"), requirement("array_record"), requirement("tensorflow")
    ],
)
