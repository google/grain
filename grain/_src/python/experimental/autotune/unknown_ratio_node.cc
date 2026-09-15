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

#include "grain/_src/python/experimental/autotune/unknown_ratio_node.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"
#include "grain/_src/python/experimental/autotune/recursive_least_squares.h"

namespace grain::autotune {

UnknownRatioNode::UnknownRatioNode(double initial_ratio, std::string name,
                                   double forgetting_factor)
    : AutotuneNode(name.empty()
                       ? absl::StrFormat("UnknownRatioNode(%f)", initial_ratio)
                       : name,
                   forgetting_factor),
      input_ratio_estimator_(RecursiveLeastSquares<1>::WithEstimate(
          initial_ratio, forgetting_factor)) {}

double UnknownRatioNode::GetInputRatio() const {
  absl::MutexLock lock(&timing_stats_mutex_);
  return input_ratio_estimator_.GetEstimate();
}

double UnknownRatioNode::GetOutputTimeMs() const {
  double input_time_ms = 0.0;
  for (const auto& input : inputs_) {
    input_time_ms += input->GetOutputTimeMs();
  }
  return GetSelfTimeMs() + input_time_ms * GetInputRatio();
}

void UnknownRatioNode::RecordRatio(double ratio) {
  absl::MutexLock lock_timing(&timing_stats_mutex_);
  (void)input_ratio_estimator_.Update(ratio);
}

absl::StatusOr<absl::Duration> UnknownRatioNode::RecordEnd() {
  double ratio = timer_.NumPauses();
  RecordRatio(ratio);
  return AutotuneNode::RecordEnd();
}

std::shared_ptr<AutotuneNode> UnknownRatioNode::Copy() const {
  double forgetting_factor;
  double input_ratio;
  {
    absl::MutexLock lock(&timing_stats_mutex_);
    forgetting_factor =
        timing_stats_.mean_latency_estimator.GetForgettingFactor();
    input_ratio = input_ratio_estimator_.GetEstimate();
  }
  return std::make_shared<UnknownRatioNode>(input_ratio, name_,
                                            forgetting_factor);
}

AutotuneNodeSnapshot UnknownRatioNode::GetSnapshotProto() const {
  AutotuneNodeSnapshot snapshot = AutotuneNode::GetSnapshotProto();
  auto* unknown_ratio = snapshot.mutable_unknown_ratio_node();
  unknown_ratio->set_current_ratio(GetInputRatio());
  return snapshot;
}

AutotuneNode::PartialDerivatives UnknownRatioNode::ComputePartialDerivatives()
    const {
  PartialDerivatives grads{
      .wrt_inputs = std::vector<double>(inputs_.size(), GetInputRatio()),
      .wrt_params = {},
  };
  return grads;
}

void UnknownRatioNode::CopyStatsTo(AutotuneNode* dest) const {
  AutotuneNode::CopyStatsTo(dest);
  UnknownRatioNode* dest_node = dynamic_cast<UnknownRatioNode*>(dest);
  if (dest_node != nullptr) {
    RecursiveLeastSquares<1> temp_estimator;
    {
      absl::MutexLock src_lock(&timing_stats_mutex_);
      temp_estimator = input_ratio_estimator_;
    }
    {
      absl::MutexLock dest_lock(&dest_node->timing_stats_mutex_);
      dest_node->input_ratio_estimator_ = temp_estimator;
    }
  }
}

}  // namespace grain::autotune
