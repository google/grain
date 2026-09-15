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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_BFGS_OPTIMIZER_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_BFGS_OPTIMIZER_H_

#include "absl/status/statusor.h"
#include "Eigen/Core"
#include "grain/_src/python/experimental/autotune/optimizer.h"

namespace grain::autotune {

// The Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm is an iterative
// method for solving unconstrained nonlinear optimization problems.
//
// See:
// https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
//
// This implementation uses the strong Wolfe condition line search to compute
// the step size at each iteration. See
// https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods#Wolfe_Conditions.
class BfgsOptimizer : public Optimizer {
 public:
  absl::StatusOr<Eigen::VectorXd> Optimize(
      const OptimizationProblem& objective) override;

 private:
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_BFGS_OPTIMIZER_H_
