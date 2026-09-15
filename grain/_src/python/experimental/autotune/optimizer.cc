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

#include "grain/_src/python/experimental/autotune/optimizer.h"

#include <algorithm>

#include "Eigen/Core"

namespace grain::autotune {
double Optimizer::OptimizationProblem::ComputeLagrangian(
    const Eigen::VectorXd& x, const Eigen::VectorXd& lambda, double rho) const {
  const int num_constraints = inequality_constraints.size();
  double penalty_sum = 0.0;
  for (int i = 0; i < num_constraints; ++i) {
    double c_val = inequality_constraints[i].c(x);
    double lambda_i = lambda[i];
    double lambda_eff = std::max(0.0, lambda_i + rho * c_val);
    // Note: this is standard ALM and expands to lambda_i * c(x) + 0.5 * rho *
    // c(x)^2. Note that there is term even when c(x) <= 0, but it is a constant
    // and does not affect the gradient.
    penalty_sum += (lambda_eff * lambda_eff - lambda_i * lambda_i);
  }
  return fn(x) + penalty_sum / (2.0 * rho);
}

Eigen::VectorXd Optimizer::OptimizationProblem::ComputeLagrangianGradient(
    Eigen::Ref<const Eigen::VectorXd> x,
    Eigen::Ref<const Eigen::VectorXd> lambda, double rho) const {
  Eigen::VectorXd constraint_gradient = Eigen::VectorXd::Zero(x.size());
  const int num_constraints = inequality_constraints.size();
  for (int i = 0; i < num_constraints; ++i) {
    double val = inequality_constraints[i].c(x);
    double lambda_eff = lambda[i] + rho * val;
    // Only accumulate active constraints.
    if (lambda_eff > 0.0) {
      constraint_gradient += lambda_eff * inequality_constraints[i].grad_c(x);
    }
  }
  return grad_fn(x) + constraint_gradient;
}

Eigen::VectorXd Optimizer::OptimizationProblem::UpdateLagrangianMultipliers(
    Eigen::Ref<const Eigen::VectorXd> x,
    Eigen::Ref<const Eigen::VectorXd> lambda, double rho) const {
  return (lambda + rho * Eigen::VectorXd::NullaryExpr(
                             lambda.size(),
                             [this, &x](int i) {
                               return inequality_constraints[i].c(x);
                             }))
      .cwiseMax(0.0);
}

}  // namespace grain::autotune
