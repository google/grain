# Fast proto parsers (serialized proto -> numpy arrays).

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

pybind_extension(
    name = "fast_proto_parser",
    srcs = ["fast_proto_parser.cc"],
    deps = [
        "//perftools/profiles/proto:profile_cc_proto",
        "//third_party/absl/strings:string_view",
        "//third_party/protobuf:protobuf_lite",
        "//third_party/tensorflow/core/example:example_protos_cc",
    ],
)

py3_binary(
    name = "usage",
    srcs = ["usage.py"],
    deps = [
        ":fast_proto_parser",
        "//third_party/py/absl:app",
        "//third_party/py/memory_profiler",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "fast_proto_parser_test",
    srcs = ["fast_proto_parser_test.py"],
    srcs_version = "PY3",
    deps = [
        ":fast_proto_parser",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)
