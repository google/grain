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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_FIXED_RATIO_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_FIXED_RATIO_NODE_H_

#include <memory>
#include <string>

#include "absl/synchronization/mutex.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"

namespace grain::autotune {

// This class represents a serialized node in the pipeline, for example a batch
// operation.
class FixedRatioNode : public AutotuneNode {
 public:
  FixedRatioNode(double input_ratio = 1.0, std::string name = "",
                 double forgetting_factor = 0.99 /* forgetting factor */);

  std::shared_ptr<AutotuneNode> Copy() const override {
    double forgetting_factor;
    {
      absl::MutexLock lock(&timing_stats_mutex_);
      forgetting_factor =
          timing_stats_.mean_latency_estimator.GetForgettingFactor();
    }
    return std::make_shared<FixedRatioNode>(input_ratio_, name_,
                                            forgetting_factor);
  }

  // Overrides.
  double GetInputRatio() const override { return input_ratio_; }
  double GetOutputTimeMs() const override;
  AutotuneNodeSnapshot GetSnapshotProto() const override;

  // Gradients.
  PartialDerivatives ComputePartialDerivatives() const override;

 private:
  double input_ratio_;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_FIXED_RATIO_NODE_H_
