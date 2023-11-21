# Packing algorithms for Grain.

package(default_visibility = ["//grain:__subpackages__"])

licenses(["notice"])

cc_library(
    name = "mffd",
    srcs = ["mffd.cc"],
    hdrs = ["mffd.h"],
    deps = [
        "//base:logging",
        "//third_party/absl/algorithm:container",
        "//third_party/absl/container:flat_hash_set",
        "//third_party/absl/status",
        "//third_party/absl/strings",
        "//util/task:status",
    ],
)

cc_test(
    name = "mffd_test",
    srcs = ["mffd_test.cc"],
    deps = [
        ":mffd",
        "//testing/base/public:gunit_main",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log",
        "//third_party/absl/random",
        "//third_party/absl/strings:str_format",
        "//util/gtl:iterator_adaptors",
    ],
)
