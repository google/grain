workspace(name = "com_google_grain")

# Might be better than http_archive
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# reference to head since the absl/log is newer then the latest release (Abseil LTS 20220623.1)
http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-master",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/master.zip",
    ],
)
# array_record 0.1.0
http_archive(
    name = "com_google_array_record",
    strip_prefix = "array_record-0.1.0",
#    sha256 = "",
    urls = ["https://github.com/google/array_record/archive/v0.1.0.zip"],
)
# V3.4.0, 20210818
http_archive(
  name = "eigen3",
  sha256 = "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626",
  strip_prefix = "eigen-3.4.0",
  urls = [
      "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2",
  ],
  build_file_content =
"""
cc_library(
    name = 'eigen3',
    srcs = [],
    includes = ['.'],
    hdrs = glob(['Eigen/**', 'unsupported/Eigen/**']),
    visibility = ['//visibility:public'],
)
"""
)
# re2 2022-04-08
http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "906d0df8ff48f8d3a00a808827f009a840190f404559f649cb8e4d7143255ef9",
    strip_prefix = "re2-a276a8c738735a0fe45a6ee590fe2df69bcf4502",
    urls = ["https://github.com/google/re2/archive/a276a8c738735a0fe45a6ee590fe2df69bcf4502.zip"],  # 2022-04-08
)
# Riegeli does not cut releases, so we reference the head
http_archive(
    name = "com_google_riegeli",
    strip_prefix = "riegeli-master",
    urls = [
        "https://github.com/google/riegeli/archive/master.zip",
    ],
)
# Dependencies required by riegeli
http_archive(
    name = "org_brotli",
    sha256 = "84a9a68ada813a59db94d83ea10c54155f1d34399baf377842ff3ab9b3b3256e",
    strip_prefix = "brotli-3914999fcc1fda92e750ef9190aa6db9bf7bdb07",
    urls = ["https://github.com/google/brotli/archive/3914999fcc1fda92e750ef9190aa6db9bf7bdb07.zip"],  # 2022-11-17
)
http_archive(
    name = "net_zstd",
    build_file = "@com_google_riegeli//third_party:net_zstd.BUILD",
    sha256 = "b6c537b53356a3af3ca3e621457751fa9a6ba96daf3aebb3526ae0f610863532",
    strip_prefix = "zstd-1.4.5/lib",
    urls = ["https://github.com/facebook/zstd/archive/v1.4.5.zip"],  # 2020-05-22
)
http_archive(
    name = "lz4",
    build_file = "@com_google_riegeli//third_party:lz4.BUILD",
    sha256 = "4ec935d99aa4950eadfefbd49c9fad863185ac24c32001162c44a683ef61b580",
    strip_prefix = "lz4-1.9.3/lib",
    urls = ["https://github.com/lz4/lz4/archive/refs/tags/v1.9.3.zip"],  # 2020-11-16
)
http_archive(
    name = "crc32c",
    build_file = "@com_google_riegeli//third_party:crc32.BUILD",
    sha256 = "338f1d9d95753dc3cdd882dfb6e176bbb4b18353c29c411ebcb7b890f361722e",
    strip_prefix = "crc32c-1.1.0",
    urls = ["https://github.com/google/crc32c/archive/1.1.0.zip"],  # 2019-05-24
)

http_archive(
    name = "highwayhash",
    build_file = "@com_google_riegeli//third_party:highwayhash.BUILD",
    sha256 = "cf891e024699c82aabce528a024adbe16e529f2b4e57f954455e0bf53efae585",
    strip_prefix = "highwayhash-276dd7b4b6d330e4734b756e97ccfb1b69cc2e12",
    urls = ["https://github.com/google/highwayhash/archive/276dd7b4b6d330e4734b756e97ccfb1b69cc2e12.zip"],  # 2019-02-22
)

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.11.0",
#    sha256 = "",
    urls = ["https://github.com/tensorflow/tensorflow/archive/v2.11.0.zip"],
)

# Needed by @org_tensorflow
# 2022-09-01
#http_archive(
#    name = "bazel_skylib",
#    sha256 = "28f81e36692e1d87823623a99966b2daf85af3fdc1b40f98e37bd5294f3dd185",
#    strip_prefix = "bazel-skylib-1.3.0",
#    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/1.3.0.zip"],
#)
# Initialize TensorFlow dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)
bind(
    name = "six",
    actual = "@six_archive//:six",
)

# This import (along with the org_tensorflow archive) is necessary to provide the devtoolset-9 toolchain
load("@org_tensorflow//tensorflow/tools/toolchains/remote_config:configs.bzl", "initialize_rbe_configs")  # buildifier: disable=load-on-top

initialize_rbe_configs()