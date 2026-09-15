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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_INTERLEAVE_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_INTERLEAVE_NODE_H_

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"

namespace grain::autotune {

// Node representing an interleave operation, which averages the statistics of
// its input nodes.
class InterleaveNode : public AutotuneNode {
 public:
  InterleaveNode(std::string name = "InterleaveNode",
                 double forgetting_factor = 0.99,
                 std::variant<double, std::shared_ptr<AutotuneParameter>>
                     cycle_length = 1.0,
                 double make_iter_buffer_size = 1.0);

  double GetInputRatio() const override;

  std::shared_ptr<AutotuneNode> Copy() const override;

  double GetOutputTimeMs() const override;

  PartialDerivatives ComputePartialDerivatives() const override;

  AutotuneNodeSnapshot GetSnapshotProto() const override;

  double GetSelfMemoryUsage() const override;
  double GetMemoryUsage() const override;
  PartialDerivatives ComputeMemoryUsagePartialDerivatives() const override;

  double GetCycleLength() const;

 private:
  void GetLocalTunableParameters(
      std::vector<std::shared_ptr<AutotuneParameter>>& parameters)
      const override;

  std::variant<double, std::shared_ptr<AutotuneParameter>> cycle_length_;
  double make_iter_buffer_size_;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_INTERLEAVE_NODE_H_
