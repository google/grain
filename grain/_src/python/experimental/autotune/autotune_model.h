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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_MODEL_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_MODEL_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "grain/_src/python/experimental/autotune/autotune_model_config.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/optimizer.h"

namespace grain::autotune {
// This is the base class for autotune models. It serves as the bridge between
// the autotune node graph and the optimization pipeline.
class AutotuneModel {
 public:
  AutotuneModel(const AutotuneModelConfig model_config)
      : model_config_(model_config) {}

  virtual ~AutotuneModel() = default;

  // Attempts to find the optimal parameters for the model, i.e. the parameters
  // that minimize the output time. Updates the underlying model with the
  // optimal parameters.
  absl::Status Optimize(std::shared_ptr<AutotuneNode> output,
                        bool apply_step_damping = false);

  // Constructs the optimization problem in a manner suitable for generic
  // optimizers.
  absl::StatusOr<Optimizer::OptimizationProblem> ConstructOptimizationProblem(
      std::shared_ptr<AutotuneNode> output);

  // Optimize only if a predefined condition has been meet, for example, if a
  // node has not been optimized in a certain amount of time. Returns true if
  // the model was optimized, false otherwise.
  absl::StatusOr<bool> MaybeOptimize(
      std::shared_ptr<AutotuneNode> output);

  size_t GetOptimizationCount() const { return optimization_count_; }

 protected:
  AutotuneModelConfig model_config_;

  size_t optimization_count_ = 0;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_MODEL_H_
