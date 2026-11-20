load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

exports_files(glob(["*.array_record-*"]))

exports_files(glob(["*.tfrecord"]))

py_proto_library(
    name = "test_record_py_pb2",
    testonly = True,
    srcs = ["test_record.proto"],
)
