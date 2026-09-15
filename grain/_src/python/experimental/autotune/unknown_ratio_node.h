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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_UNKNOWN_RATIO_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_UNKNOWN_RATIO_NODE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/recursive_least_squares.h"

namespace grain::autotune {

// This class represents a node with an unknown I/O ratio in the pipeline, for
// example a filter operation. The ratio is estimated during runtime by
// recording observations.
class UnknownRatioNode : public AutotuneNode {
 public:
  UnknownRatioNode(double initial_ratio = 1.0, std::string name = "",
                   double forgetting_factor = 0.99 /* forgetting factor */);

  // Overrides.
  std::shared_ptr<AutotuneNode> Copy() const override;
  double GetInputRatio() const override;
  double GetOutputTimeMs() const override;
  AutotuneNodeSnapshot GetSnapshotProto() const override;
  absl::StatusOr<absl::Duration> RecordEnd() override;

  // Update the ratio estimation with a new observation.
  void RecordRatio(double ratio);

  // Gradients.
  PartialDerivatives ComputePartialDerivatives() const override;

  void CopyStatsTo(AutotuneNode* dest) const override;

 private:
  RecursiveLeastSquares<1> input_ratio_estimator_
      ABSL_GUARDED_BY(timing_stats_mutex_);
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_UNKNOWN_RATIO_NODE_H_
