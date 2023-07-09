workspace(name = "com_google_grain")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
    ],
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")  #buildifier: disable=load-on-top

bazel_skylib_workspace()

# `pybind11_bazel` (https://github.com/pybind/pybind11_bazel): 20230130
http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-5f458fa53870223a0de7eeb60480dd278b442698",
  sha256 = "b35f3abc3d52ee5c753fdeeb2b5129b99e796558754ca5d245e28e51c1072a21",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/5f458fa53870223a0de7eeb60480dd278b442698.tar.gz"],
)
# V2.10.3, 20230130
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.3",
  sha256 = "201966a61dc826f1b1879a24a3317a1ec9214a918c8eb035be2f30c3e9cfbdcb",
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.3.zip"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")