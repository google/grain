load("@org_tensorflow//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_strict_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

pytype_strict_library(
    name = "batching",
    srcs = ["batching.py"],
    deps = [
        ":transforms",
        "//grain/_src/core:constants",
        "//grain/_src/tensorflow/ops:batch_and_pack",
        # "Implicit dependency jax",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "batching_test",
    srcs = ["batching_test.py"],
    shard_count = 10,
    deps = [
        ":batching",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency chex",
        "//grain/_src/core:constants",
        # "Implicit dependency jax",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "checkpoint_handlers",
    srcs = ["checkpoint_handlers.py"],
    deps = [
        ":data_iterators",
        # "Implicit dependency etils/epath",
        # "Implicit dependency jax",
        # "Implicit dependency orbax/checkpoint",
    ],
)

pytype_strict_library(
    name = "data_iterators",
    srcs = ["data_iterators.py"],
    deps = [
        ":index_dataset",
        # "Implicit dependency clu/data:dataset_iterator",
        # "Implicit dependency etils/epath",
        "//grain/_src/core:constants",
        # "Implicit dependency jax",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "data_iterators_test",
    srcs = ["data_iterators_test.py"],
    deps = [
        ":data_iterators",
        ":index_dataset",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency chex",
        # "Implicit dependency clu/data:dataset_iterator",
        # "Implicit dependency etils/epath",
        "//grain/_src/core:constants",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "data_sources",
    srcs = ["data_sources.py"],
    deps = [
        # "Implicit dependency etils/epath",
        "//grain/_src/core:usage_logging",
        "//grain/_src/tensorflow/ops:array_record_data_source",
        # "Implicit dependency jax",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
        # "Implicit dependency tensorflow_datesets",
    ],
)

py_strict_test(
    name = "data_sources_test",
    srcs = ["data_sources_test.py"],
    data = [
        # "Implicit dependency tensorflow_datesets/testing/metadata:imagenet2012",
    ],
    deps = [
        ":data_sources",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency pandas",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
        # "Implicit dependency tensorflow_datesets",
    ],
)

pytype_strict_library(
    name = "data_loaders",
    srcs = ["data_loaders.py"],
    deps = [
        ":batching",
        ":data_iterators",
        ":data_sources",
        ":index_dataset",
        ":transforms",
        # "Implicit dependency absl/logging",
        # "Implicit dependency clu:preprocess_spec",
        # "Implicit dependency etils/epath",
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:sharding",
        "//grain/_src/core:usage_logging",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
        # "Implicit dependency tensorflow_datesets",
    ],
)

py_strict_test(
    name = "data_loaders_test",
    srcs = ["data_loaders_test.py"],
    deps = [
        ":data_loaders",
        # "Implicit dependency absl/testing:parameterized",
        "//grain/_src/core:constants",
        "//grain/_src/core:sharding",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
        # "Implicit dependency tensorflow_datesets",
    ],
)

pytype_strict_library(
    name = "index_dataset",
    srcs = ["index_dataset.py"],
    deps = [
        # "Implicit dependency absl/logging",
        "//grain/_src/core:config",
        "//grain/_src/core:constants",
        "//grain/_src/core:random",
        "//grain/_src/core:sharding",
        "//grain/_src/core:usage_logging",
        # "Implicit dependency jax",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "index_dataset_test",
    srcs = ["index_dataset_test.py"],
    shard_count = 20,
    deps = [
        ":index_dataset",
        # "Implicit dependency absl/testing:parameterized",
        "//grain/_src/core:constants",
        "//grain/_src/core:sharding",
        # "Implicit dependency jax",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "transforms",
    srcs = ["transforms.py"],
    deps = [
        # "Implicit dependency absl/logging",
        # "Implicit dependency clu:preprocess_spec",
        "//grain/_src/core:constants",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
        # "Implicit dependency typing_extensions",
    ],
)

py_strict_test(
    name = "transforms_test",
    srcs = ["transforms_test.py"],
    deps = [
        ":transforms",
        # "Implicit dependency absl/testing:parameterized",
        "//grain/_src/core:constants",
        # "Implicit dependency seqio",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)
