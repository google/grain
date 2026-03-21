load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

default_visibility = ["//grain:__subpackages__"]

package(default_visibility = default_visibility)

py_proto_library(
    name = "execution_summary_py_pb2",
    # For profiling tooling.
    visibility = default_visibility + [
        "//learning/gemini/cms/core/data/iterators:__subpackages__",
        "//learning/gemini/gemax/core/data/iterators:__subpackages__",
    srcs = ["execution_summary.proto"],
)
