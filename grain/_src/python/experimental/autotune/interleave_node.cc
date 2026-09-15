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

#include "grain/_src/python/experimental/autotune/interleave_node.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"

namespace grain::autotune {

InterleaveNode::InterleaveNode(
    std::string name, double forgetting_factor,
    std::variant<double, std::shared_ptr<AutotuneParameter>> cycle_length,
    double make_iter_buffer_size)
    : AutotuneNode(std::move(name), forgetting_factor),
      cycle_length_(cycle_length),
      make_iter_buffer_size_(make_iter_buffer_size) {}

double InterleaveNode::GetInputRatio() const { return 1.0 / GetCycleLength(); }

std::shared_ptr<AutotuneNode> InterleaveNode::Copy() const {
  absl::MutexLock lock(&timing_stats_mutex_);
  return std::make_shared<InterleaveNode>(
      name_, timing_stats_.mean_latency_estimator.GetForgettingFactor(),
      cycle_length_, make_iter_buffer_size_);
}

double InterleaveNode::GetOutputTimeMs() const {
  double input_output_time_ms = 0.0;
  for (const auto& input : inputs_) {
    input_output_time_ms += input->GetOutputTimeMs();
  }
  return GetSelfTimeMs() + input_output_time_ms * GetInputRatio();
}

AutotuneNode::PartialDerivatives InterleaveNode::ComputePartialDerivatives()
    const {
  double input_output_time_ms = 0.0;
  for (const auto& input : inputs_) {
    input_output_time_ms += input->GetOutputTimeMs();
  }

  double cycle_length = GetCycleLength();
  double d_tout_d_c = -input_output_time_ms / (cycle_length * cycle_length);

  PartialDerivatives grads{
      .wrt_inputs = std::vector<double>(inputs_.size(), GetInputRatio()),
      .wrt_params = {},
  };

  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(
          cycle_length_)) {
    auto param = std::get<std::shared_ptr<AutotuneParameter>>(cycle_length_);
    grads.wrt_params[param->id()] = d_tout_d_c;
  }

  return grads;
}

AutotuneNodeSnapshot InterleaveNode::GetSnapshotProto() const {
  AutotuneNodeSnapshot snapshot = AutotuneNode::GetSnapshotProto();
  snapshot.mutable_interleave_node()->set_cycle_length(
      static_cast<int32_t>(GetCycleLength()));
  return snapshot;
}

double InterleaveNode::GetSelfMemoryUsage() const { return 0.0; }

double InterleaveNode::GetMemoryUsage() const {
  if (inputs_.empty()) return 0.0;
  double sum_input_memory = 0;
  for (const auto& input : inputs_) {
    sum_input_memory += input->GetMemoryUsage();
  }
  double avg_input_memory = sum_input_memory / inputs_.size();
  return (GetCycleLength() + make_iter_buffer_size_) * avg_input_memory;
}

AutotuneNode::PartialDerivatives
InterleaveNode::ComputeMemoryUsagePartialDerivatives() const {
  absl::flat_hash_map<int64_t, double> parameter_gradients;

  double dM_dc = 0;
  double avg_input_memory = 0;
  if (!inputs_.empty()) {
    double sum_input_memory = 0;
    for (const auto& input : inputs_) {
      sum_input_memory += input->GetMemoryUsage();
    }
    avg_input_memory = sum_input_memory / inputs_.size();
    dM_dc = avg_input_memory;
  }

  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(
          cycle_length_)) {
    auto param = std::get<std::shared_ptr<AutotuneParameter>>(cycle_length_);
    parameter_gradients[param->id()] = dM_dc;
  }

  return PartialDerivatives{
      .wrt_inputs = std::vector<double>(inputs_.size(), 1.0),
      .wrt_params = std::move(parameter_gradients),
  };
}

double InterleaveNode::GetCycleLength() const {
  double val;
  if (std::holds_alternative<double>(cycle_length_)) {
    val = std::get<double>(cycle_length_);
  } else {
    val =
        std::get<std::shared_ptr<AutotuneParameter>>(cycle_length_)->GetValue();
  }
  return std::max(1.0, val);
}

void InterleaveNode::GetLocalTunableParameters(
    std::vector<std::shared_ptr<AutotuneParameter>>& parameters) const {
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(
          cycle_length_)) {
    parameters.push_back(
        std::get<std::shared_ptr<AutotuneParameter>>(cycle_length_));
  }
}

}  // namespace grain::autotune
