// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_SERIALIZATION_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_SERIALIZATION_H_

#include <memory>

#include "absl/status/statusor.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"

namespace grain::autotune {

// Reconstructs the node and its children from a proto snapshot.
// This is a free function to avoid circular dependencies between AutotuneNode
// and its subclasses.
absl::StatusOr<std::shared_ptr<AutotuneNode>> RecoverAutotuneNode(
    const AutotuneNodeSnapshot& snapshot);

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_SERIALIZATION_H_
