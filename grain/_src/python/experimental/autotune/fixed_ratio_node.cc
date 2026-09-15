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

#include "grain/_src/python/experimental/autotune/fixed_ratio_node.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"

namespace grain::autotune {

FixedRatioNode::FixedRatioNode(double input_ratio, std::string name,
                               double forgetting_factor)
    : AutotuneNode(name.empty()
                       ? absl::StrFormat("FixedRatioNode(%f)", input_ratio)
                       : name,
                   forgetting_factor),
      input_ratio_(input_ratio) {}

double FixedRatioNode::GetOutputTimeMs() const {
  double input_time_ms = 0.0;
  for (const auto& input : inputs_) {
    input_time_ms += input->GetOutputTimeMs();
  }
  return GetSelfTimeMs() + input_time_ms * input_ratio_;
}

AutotuneNodeSnapshot FixedRatioNode::GetSnapshotProto() const {
  AutotuneNodeSnapshot snapshot = AutotuneNode::GetSnapshotProto();
  auto* fixed_ratio = snapshot.mutable_fixed_ratio_node();
  fixed_ratio->set_input_ratio(input_ratio_);
  return snapshot;
}

AutotuneNode::PartialDerivatives FixedRatioNode::ComputePartialDerivatives()
    const {
  PartialDerivatives grads{
      .wrt_inputs = std::vector<double>(inputs_.size(), GetInputRatio()),
      .wrt_params = {},
  };
  return grads;
}

}  // namespace grain::autotune
