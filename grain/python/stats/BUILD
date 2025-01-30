load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

default_visibility = ["//grain:__subpackages__"]

package(default_visibility = default_visibility)

proto_library(
    name = "execution_summary_proto",
    srcs = ["execution_summary.proto"],
    # For profiling tooling.
)

py_proto_library(
    name = "execution_summary_py_pb2",
    # For profiling tooling.
    deps = [":execution_summary_proto"],
)
