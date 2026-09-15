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

#include "grain/_src/python/experimental/autotune/autotune_model.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "grain/_src/python/experimental/autotune/autotune_model_config.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"
#include "grain/_src/python/experimental/autotune/bfgs_optimizer.h"
#include "grain/_src/python/experimental/autotune/optimizer.h"

namespace grain::autotune {

constexpr int kMinStatsSteps = 5;

absl::Status AutotuneModel::Optimize(
    std::shared_ptr<AutotuneNode> output,
    bool apply_step_damping) {
  auto snapshot = output->GetSnapshot();
  auto problem_or = ConstructOptimizationProblem(snapshot);
  if (!problem_or.ok()) {
    return problem_or.status();
  }
  Optimizer::OptimizationProblem problem = std::move(problem_or.value());
  std::vector<std::shared_ptr<AutotuneParameter>> params =
      snapshot->GetTunableParameters();
  Eigen::VectorXd grad0 = problem.grad_fn(problem.x0);

  BfgsOptimizer optimizer;
  auto x_min_or = optimizer.Optimize(problem);
  Eigen::VectorXd x_min;
  if (x_min_or.ok()) {
    x_min = x_min_or.value();
  } else {
    VLOG(1) << "Optimization failed: " << x_min_or.status();
    x_min = problem.x0;
  }

  std::vector<std::shared_ptr<AutotuneParameter>> live_params =
      output->GetTunableParameters();

  for (int i = 0; i < params.size(); ++i) {
    VLOG(1) << "Autotune optimization for node " << output->GetName()
            << ", parameter " << params[i]->GetName()
            << ": initial_value=" << problem.x0[i] << ", gradient=" << grad0[i]
            << ", proposed_value=" << x_min[i]
            << ", final_value=" << std::round(x_min[i]);
  }
  for (const auto& in : output->GetInputs()) {
    VLOG(2) << "Input node: " << in->GetName()
            << " BaseThroughput=" << in->GetBaseThroughput()
            << " Contention=" << in->GetContention()
            << " Coherency=" << in->GetCoherency();
  }

  // Round to the nearest integer and clamp to the parameter's valid range.
  for (int i = 0; i < live_params.size(); ++i) {
    auto [lower, upper] = live_params[i]->GetRange();
    double current_val = problem.x0[i];
    double target_val = x_min[i];
    double next_val = target_val;
    if (apply_step_damping) {
      // Apply trust-region damping during online tuning to prevent massive
      // parameter jumps before the USL model achieves multi-point spectral
      // conditioning.
      double max_step = std::max(2.0, current_val * 0.5);
      double step =
          std::max(-max_step, std::min(max_step, target_val - current_val));
      next_val = current_val + step;
    }
    double rounded_val = std::round(next_val);
    double clamped_val = std::max(lower, std::min(upper, rounded_val));
    auto status =
        live_params[i]->SetValue(clamped_val, /*validate_value=*/true);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Optimizer::OptimizationProblem>
AutotuneModel::ConstructOptimizationProblem(
    std::shared_ptr<AutotuneNode> output) {
  std::vector<std::shared_ptr<AutotuneParameter>> params =
      output->GetTunableParameters();
  Eigen::VectorXd x0(params.size());
  for (int i = 0; i < params.size(); ++i) {
    x0[i] = params[i]->GetValue();
  }
  auto f = [output, params](Eigen::Ref<const Eigen::VectorXd> x) -> double {
    for (int i = 0; i < params.size(); ++i) {
      params[i]->SetValue(x[i], false).IgnoreError();
    }
    return output->GetOutputTimeMs();
  };
  auto grad_f =
      [output, params](Eigen::Ref<const Eigen::VectorXd> x) -> Eigen::VectorXd {
    Eigen::VectorXd grad(x.size());
    for (int i = 0; i < params.size(); ++i) {
      params[i]->SetValue(x[i], false).IgnoreError();
    }
    auto grads = output->ComputeParameterGradients();
    for (size_t i = 0; i < params.size(); ++i) {
      double sensitivity = grads[params[i]->id()];
      if (!std::isfinite(sensitivity)) {
        std::cerr << "NON-FINITE SENSITIVITY for " << params[i]->GetName()
                  << ": " << sensitivity << std::endl;
        grad[i] = std::numeric_limits<double>::quiet_NaN();
      } else {
        grad[i] = sensitivity;
      }
    }
    return grad;
  };

  std::vector<Optimizer::InequalityConstraint> inequality_constraints;
  // Add the local constraints for each parameter.
  for (int i = 0; i < params.size(); ++i) {
    const AutotuneParameter& param = *params[i];
    // Trust region constraints to force systematic exploration (sweep).
    auto [lower_bound, upper_bound] = param.GetRange();
    double multiplier = param.GetTrustRegionMultiplier();
    // The optimizer enforces that c(x) <= 0 for all constraints.
    // For small initial values, ensure we can at least move to the next/prev
    // integer by setting a minimum trust region size.
    lower_bound = std::max(lower_bound, x0[i] / multiplier);
    auto c_lower = [lower_bound,
                    i](Eigen::Ref<const Eigen::VectorXd> x) -> double {
      return lower_bound - x[i];
    };
    auto grad_c_lower =
        [i](Eigen::Ref<const Eigen::VectorXd> x) -> Eigen::VectorXd {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
      grad[i] = -1.0;
      return grad;
    };
    inequality_constraints.push_back(
        {.c = std::move(c_lower), .grad_c = std::move(grad_c_lower)});
    upper_bound = std::min(upper_bound, x0[i] * multiplier);
    auto c_upper = [upper_bound,
                    i](Eigen::Ref<const Eigen::VectorXd> x) -> double {
      return x[i] - upper_bound;
    };
    auto grad_c_upper =
        [i](Eigen::Ref<const Eigen::VectorXd> x) -> Eigen::VectorXd {
      Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
      grad[i] = 1.0;
      return grad;
    };
    inequality_constraints.push_back(
        {.c = std::move(c_upper), .grad_c = std::move(grad_c_upper)});
  }

  // Add the global constraint for RAM usage.
  auto ram_budget_or = model_config_.GetRamBudgetGb();
  if (!ram_budget_or.ok()) {
    return ram_budget_or.status();
  }
  double ram_budget_gb = ram_budget_or.value();
  const double kBytesInGb = 1024.0 * 1024.0 * 1024.0;
  inequality_constraints.push_back(
      {.c = [output, params, kBytesInGb,
             ram_budget_gb](Eigen::Ref<const Eigen::VectorXd> x) -> double {
         for (int i = 0; i < params.size(); ++i) {
           params[i]->SetValue(x[i], false).IgnoreError();
         }
         return output->GetMemoryUsage() / kBytesInGb - ram_budget_gb;
       },
       .grad_c = [output, params, kBytesInGb](
                     Eigen::Ref<const Eigen::VectorXd> x) -> Eigen::VectorXd {
         for (int i = 0; i < params.size(); ++i) {
           params[i]->SetValue(x[i], false).IgnoreError();
         }
         absl::flat_hash_map<int64_t, double> derivatives =
             output->ComputeMemoryUsageGradients();
         Eigen::VectorXd grad(params.size());
         for (int i = 0; i < params.size(); ++i) {
           grad[i] = derivatives[params[i]->id()] / kBytesInGb;
         }
         return grad;
       }});

  return Optimizer::OptimizationProblem{
      .fn = std::move(f),
      .grad_fn = std::move(grad_f),
      .x0 = std::move(x0),
      .inequality_constraints = std::move(inequality_constraints),
  };
}

absl::StatusOr<bool> AutotuneModel::MaybeOptimize(
    std::shared_ptr<AutotuneNode> output) {
  optimization_count_++;
  // Do not optimize until we have collected some stats after warmup.
  if (optimization_count_ < kMinStatsSteps) {
    return false;
  }
  if (optimization_count_ % model_config_.GetOptimizationFrequency() == 0) {
    VLOG(3) << "Optimizing node: " << output->GetName()
            << ", optimization count = " << optimization_count_;
    auto status = Optimize(output, /*apply_step_damping=*/true);
    if (!status.ok()) {
      return status;
    }
    return true;
  }
  return false;
}

}  // namespace grain::autotune
