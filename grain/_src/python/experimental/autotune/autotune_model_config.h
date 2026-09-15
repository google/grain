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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_MODEL_CONSTRAINTS_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_MODEL_CONSTRAINTS_H_

#include <cstddef>
#include <optional>
#include <string>

#include "absl/status/statusor.h"

namespace grain::autotune {

// Model constraints for the autotune pipeline either estimates the available
// CPU and RAM or respectively uses user defined CPU and RAM budgets for the
// autotune pipeline.

class AutotuneModelConfig {
 public:
  // 3p version which enforces providing the CPU and RAM budgets.
  explicit AutotuneModelConfig(
      std::optional<size_t> cpu_budget = std::nullopt,
      std::optional<double> ram_budget_gb = std::nullopt,
      int optimization_frequency = 100, int warmup_steps = 0);
  virtual ~AutotuneModelConfig() = default;
  // Returns the CPU budget. If not set, it will be estimated.
  absl::StatusOr<size_t> GetCpuBudget();
  // Returns the RAM budget in GB. If not set, it will be estimated.
  absl::StatusOr<double> GetRamBudgetGb();
  // Returns the optimization frequency.
  int GetOptimizationFrequency() const { return optimization_frequency_; }
  // Returns the number of initial steps needed to skip optimization and
  // parameter learning.
  int GetWarmupSteps() const { return warmup_steps_; }

 protected:
  // Reads the content of a file into a string. Virtual for testing.
  virtual std::string ReadFileToString(const std::string& path) const;

  // Estimates resource availability by checking Linux control groups (Cgroups).
  //
  // Cgroup v1 (legacy): Uses separate hierarchies for different resource
  // controllers (e.g., cpu, memory), typically mounted under
  // /sys/fs/cgroup/<controller>.
  //
  // Cgroup v2 (unified): Provides a newer, simplified hierarchy where all
  // controllers are managed under a single tree, typically mounted at
  // /sys/fs/cgroup.
  absl::StatusOr<double> EstimateRamAvailabilityCgroupV2() const;
  absl::StatusOr<size_t> EstimateCpuAvailabilityCgroupV2() const;
  absl::StatusOr<double> EstimateRamAvailabilityCgroupV1() const;
  absl::StatusOr<size_t> EstimateCpuAvailabilityCgroupV1() const;

 private:
  absl::StatusOr<size_t> EstimateCpuAvailability() const;
  absl::StatusOr<double> EstimateRamAvailabilityGb() const;

  std::optional<size_t> cpu_budget_;
  std::optional<double> ram_budget_gb_;
  int optimization_frequency_;
  int warmup_steps_;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_MODEL_CONSTRAINTS_H_
