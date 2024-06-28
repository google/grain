# Fast proto parsers (serialized proto -> numpy arrays).

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = ["//grain:__subpackages__"])

pybind_extension(
    name = "encode",
    srcs = ["encode.cc"],
    deps = [
        "//third_party/absl/strings",
        "//third_party/protobuf:protobuf_lite",
        "//third_party/py/numpy:headers",
        "//third_party/pybind11_protobuf:native_proto_caster",
        "//third_party/tensorflow/core/example:example_protos_cc",
    ],
)

py_test(
    name = "encode_test",
    srcs = ["encode_test.py"],
    srcs_version = "PY3",
    deps = [
        ":encode",
    ],
)

pybind_extension(
    name = "decode",
    srcs = ["decode.cc"],
    deps = [
        "//third_party/absl/strings:string_view",
        "//third_party/absl/types:span",
        "//third_party/protobuf:protobuf_lite",
        "//third_party/pybind11_protobuf:native_proto_caster",
        "//third_party/tensorflow/core/example:example_protos_cc",
    ],
)

py3_binary(
    name = "usage",
    srcs = ["usage.py"],
    deps = [
        ":decode",
        ":encode",
        "//third_party/py/absl:app",
        "//third_party/py/memory_profiler",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "decode_test",
    srcs = ["decode_test.py"],
    srcs_version = "PY3",
    deps = [
        ":decode",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)
