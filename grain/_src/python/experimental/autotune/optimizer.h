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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_OPTIMIZER_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_OPTIMIZER_H_

#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "Eigen/Core"

namespace grain::autotune {

class Optimizer {
 public:
  using Function =
      absl::AnyInvocable<double(Eigen::Ref<const Eigen::VectorXd>) const>;
  using GradientFunction = absl::AnyInvocable<Eigen::VectorXd(
      Eigen::Ref<const Eigen::VectorXd>) const>;

  // Constraint function, imposes c(x) <= 0.
  struct InequalityConstraint {
    Function c;
    GradientFunction grad_c;
  };
  // Abstraction of the optimization problem suitable for generic optimizers.
  struct OptimizationProblem {
    // Updates the augmented Lagrangian multipliers.
    Eigen::VectorXd ComputeAugmentedLagrangianMultipliers(
        Eigen::Ref<const Eigen::VectorXd> x,
        Eigen::Ref<const Eigen::VectorXd> lambda, double rho) const;

    // Contributions of the constraints to the objective function.
    double ComputeLagrangian(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& lambda, double rho) const;

    Eigen::VectorXd UpdateLagrangianMultipliers(
        Eigen::Ref<const Eigen::VectorXd> x,
        Eigen::Ref<const Eigen::VectorXd> lambda, double rho) const;

    // Contributions of the constraints to the gradient of the objective
    // function.
    Eigen::VectorXd ComputeLagrangianGradient(
        Eigen::Ref<const Eigen::VectorXd> x,
        Eigen::Ref<const Eigen::VectorXd> lambda, double rho) const;

    Function fn;
    GradientFunction grad_fn;
    Eigen::VectorXd x0;
    std::vector<InequalityConstraint> inequality_constraints;
  };

  virtual ~Optimizer() = default;

  virtual absl::StatusOr<Eigen::VectorXd> Optimize(
      const OptimizationProblem& objective) = 0;

 private:
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_OPTIMIZER_H_
