load("@org_tensorflow//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "py_strict_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_wrapper_py", "tf_py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tf_custom_op_library(
    name = "array_record_ops.so",
    srcs = ["array_record_ops.cc"],
    deps = [
        # "Remove dependency //file/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_array_record//cpp:array_record_reader",
        "@com_google_array_record//cpp:thread_pool",
        # "Remove dependency google/placer",
        "@com_googlesource_code_re2//:re2",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/core/platform:errors",
    ],
)

pytype_strict_library(
    name = "array_record_data_source",
    srcs = ["array_record_data_source.py"],
    data = ["array_record_ops.so"],
    deps = [
        # "Implicit dependency etils/epath",
        "//grain/_src/core:usage_logging",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

tf_py_test(
    name = "array_record_data_source_test",
    srcs = ["array_record_data_source_test.py"],
    data = [
        "//grain/_src/tensorflow/ops/testdata:alphabet.array_record-00000-of-00002",
        "//grain/_src/tensorflow/ops/testdata:alphabet.array_record-00001-of-00002",
    ],
    deps = [
        ":array_record_data_source",
        # "Implicit dependency absl/flags",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

tf_custom_op_library(
    name = "batch_and_pack_op.so",
    srcs = ["batch_and_pack_op.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/data:dataset_utils",
        "@org_tensorflow//tensorflow/core/data:name_utils",
        "@org_tensorflow//tensorflow/core/kernels:fill_functor",
        "@org_tensorflow//tensorflow/core/platform:blocking_counter",
        "@org_tensorflow//tensorflow/core/platform:macros",
    ],
)

pytype_strict_library(
    name = "batch_and_pack",
    srcs = ["batch_and_pack.py"],
    data = ["batch_and_pack_op.so"],
    deps = [
        # "Implicit dependency absl:app",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)

tf_py_test(
    name = "batch_and_pack_test",
    srcs = ["batch_and_pack_test.py"],
    deps = [
        ":batch_and_pack",
        # "Implicit dependency absl/testing:parameterized",
        # "Implicit dependency numpy",
        # "Implicit dependency tensorflow:tensorflow_no_contrib",
    ],
)
