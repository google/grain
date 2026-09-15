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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOURCE_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOURCE_NODE_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"

namespace grain::autotune {

class SourceNode : public AutotuneNode {
 public:
  SourceNode(std::string name = "SourceNode",
             double forgetting_factor = 0.99 /* forgetting factor */)
      : AutotuneNode(name, forgetting_factor) {}

  std::shared_ptr<AutotuneNode> Copy() const override {
    double forgetting_factor;
    {
      absl::MutexLock lock(&timing_stats_mutex_);
      forgetting_factor =
          timing_stats_.mean_latency_estimator.GetForgettingFactor();
    }
    return std::make_shared<SourceNode>(name_, forgetting_factor);
  }

  absl::StatusOr<std::shared_ptr<AutotuneNode>> AddInput(
      std::shared_ptr<AutotuneNode> input) override {
    return absl::InvalidArgumentError("Source node cannot have inputs.");
  }

  double GetInputRatio() const override { return 0.0; }

  PartialDerivatives ComputePartialDerivatives() const override {
    // No gradients because this is currently treated as constant. Eventually if
    // we want to add tunable parameters for filesystem reads for example we can
    // consider including them here presuming we have an appropriate timing
    // model. Sources by definition have no inputs.
    return PartialDerivatives{};
  }

  AutotuneNodeSnapshot GetSnapshotProto() const override {
    AutotuneNodeSnapshot snapshot = AutotuneNode::GetSnapshotProto();
    snapshot.mutable_source_node();
    return snapshot;
  }
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOURCE_NODE_H_
