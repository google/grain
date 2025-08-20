load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

default_visibility = ["//grain:__subpackages__"]

package(default_visibility = default_visibility)

py_proto_library(
    name = "execution_summary_py_pb2",
    # For profiling tooling.
    srcs = ["execution_summary.proto"],
)
