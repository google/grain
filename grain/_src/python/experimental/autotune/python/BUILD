# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//grain:__subpackages__"])

pybind_extension(
    name = "bindings",
    srcs = ["bindings.cc"],
    linkopts = select({
        "@platforms//os:osx": ["-Wl,-exported_symbol,_PyInit_bindings"],
        "@platforms//os:windows": [],
        "//conditions:default": ["-Wl,--exclude-libs,ALL"],
    }),
    deps = [
        "@abseil-cpp//absl/strings:str_format",
        "//grain/_src/python/experimental/autotune:async_fixed_ratio_node",
        "//grain/_src/python/experimental/autotune:autotune_model",
        "//grain/_src/python/experimental/autotune:autotune_model_config",
        "//grain/_src/python/experimental/autotune:autotune_node",
        "//grain/_src/python/experimental/autotune:autotune_node_cc_proto",
        "//grain/_src/python/experimental/autotune:autotune_parameter",
        "//grain/_src/python/experimental/autotune:autotune_serialization",
        "//grain/_src/python/experimental/autotune:fixed_ratio_node",
        "//grain/_src/python/experimental/autotune:interleave_node",
        "//grain/_src/python/experimental/autotune:source_node",
        "//grain/_src/python/experimental/autotune:unknown_ratio_node",
    ],
)

py_test(
    name = "bindings_test",
    srcs = ["bindings_test.py"],
    srcs_version = "PY3",
    data = select({
        "@platforms//os:windows": [":bindings.pyd"],
        "//conditions:default": [":bindings.so"],
    }),
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":bindings",
        "@abseil-py//absl/testing:absltest",
        "@abseil-py//absl/testing:parameterized",
    ],
)

py_test(
    name = "autotune_serialization_test",
    srcs = ["autotune_serialization_test.py"],
    srcs_version = "PY3",
    data = select({
        "@platforms//os:windows": [":bindings.pyd"],
        "//conditions:default": [":bindings.so"],
    }),
    target_compatible_with = select({
        "@platforms//os:windows": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    deps = [
        ":bindings",
        "@abseil-py//absl/testing:absltest",
    ],
)
