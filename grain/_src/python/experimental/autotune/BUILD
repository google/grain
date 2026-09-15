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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//grain:__subpackages__"])

proto_library(
    name = "autotune_node_proto",
    srcs = ["autotune_node.proto"],
)

cc_proto_library(
    name = "autotune_node_cc_proto",
    deps = [":autotune_node_proto"],
)

cc_library(
    name = "softplus",
    srcs = ["softplus.cc"],
    hdrs = ["softplus.h"],
)

cc_library(
    name = "autotune_node",
    srcs = ["autotune_node.cc"],
    hdrs = ["autotune_node.h"],
    deps = [
        ":autotune_node_cc_proto",
        ":autotune_parameter",
        ":recursive_least_squares",
        ":softplus",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@eigen",
    ],
)

cc_library(
    name = "fixed_ratio_node",
    srcs = ["fixed_ratio_node.cc"],
    hdrs = ["fixed_ratio_node.h"],
    deps = [
        ":autotune_node",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "unknown_ratio_node",
    srcs = ["unknown_ratio_node.cc"],
    hdrs = ["unknown_ratio_node.h"],
    deps = [
        ":autotune_node",
        ":autotune_node_cc_proto",
        ":recursive_least_squares",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "source_node",
    hdrs = ["source_node.h"],
    deps = [
        ":autotune_node",
        ":autotune_node_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "interleave_node",
    srcs = ["interleave_node.cc"],
    hdrs = ["interleave_node.h"],
    deps = [
        ":autotune_node",
        ":autotune_node_cc_proto",
        ":autotune_parameter",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "autotune_parameter",
    srcs = ["autotune_parameter.cc"],
    hdrs = ["autotune_parameter.h"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "async_fixed_ratio_node",
    srcs = ["async_fixed_ratio_node.cc"],
    hdrs = ["async_fixed_ratio_node.h"],
    deps = [
        ":autotune_node",
        ":autotune_node_cc_proto",
        ":autotune_parameter",
        ":softplus",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "recursive_least_squares",
    hdrs = ["recursive_least_squares.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@eigen",
    ],
)

cc_library(
    name = "optimizer",
    srcs = ["optimizer.cc"],
    hdrs = ["optimizer.h"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "bfgs_optimizer",
    srcs = ["bfgs_optimizer.cc"],
    hdrs = ["bfgs_optimizer.h"],
    deps = [
        ":optimizer",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@eigen",
    ],
)

cc_library(
    name = "autotune_model_config",
    srcs = ["autotune_model_config.cc"],
    hdrs = ["autotune_model_config.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "autotune_model",
    srcs = ["autotune_model.cc"],
    hdrs = ["autotune_model.h"],
    deps = [
        ":autotune_model_config",
        ":autotune_node",
        ":autotune_parameter",
        ":bfgs_optimizer",
        ":optimizer",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen",
    ],
)

cc_library(
    name = "autotune_serialization",
    srcs = ["autotune_serialization.cc"],
    hdrs = ["autotune_serialization.h"],
    deps = [
        ":async_fixed_ratio_node",
        ":autotune_node",
        ":autotune_node_cc_proto",
        ":fixed_ratio_node",
        ":interleave_node",
        ":recursive_least_squares",
        ":source_node",
        ":unknown_ratio_node",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)
