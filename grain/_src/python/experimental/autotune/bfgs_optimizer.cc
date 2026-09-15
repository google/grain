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

#include "grain/_src/python/experimental/autotune/bfgs_optimizer.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "grain/_src/python/experimental/autotune/optimizer.h"

namespace grain::autotune {
namespace {

// This version of the line search ensures that the Armijo condition and the
// curvature condition are satisfied. See
// https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods#Wolfe_Conditions.
double StrongWolfeConditionLineSearch(const Optimizer::OptimizationProblem& obj,
                                      Eigen::Ref<const Eigen::VectorXd> x,
                                      Eigen::Ref<const Eigen::VectorXd> p,
                                      Eigen::Ref<const Eigen::VectorXd> lambda,
                                      double rho) {
  constexpr int kMaxIters = 100;
  constexpr double kC1 = 1e-4;
  constexpr double kC2 = 0.9;
  double alpha_min = 0.0;
  double alpha_max = std::numeric_limits<double>::infinity();
  double alpha = 1.0;
  std::optional<double> last_valid_alpha = std::nullopt;
  double fx = obj.ComputeLagrangian(x, lambda, rho);
  Eigen::VectorXd gx = obj.ComputeLagrangianGradient(x, lambda, rho);
  const double g_dot_p = gx.dot(p);

  // Check for a bad direction. This indicates the Hessian approximation is
  // poorly conditioned and must be reset.
  if (g_dot_p > 0) {
    return 0.0;
  }

  for (int i = 0; i < kMaxIters; ++i) {
    Eigen::VectorXd x_new = x + alpha * p;
    double fx_new = obj.ComputeLagrangian(x_new, lambda, rho);

    // Check that the Armijo condition is satisfied. Armijo condition checks
    // that the objective function is sufficiently decreased. See
    // https://optimization.cbe.cornell.edu/index.php?title=Line_search_methods#Armijo_(Sufficient_Decrease)_Condition
    if (fx_new > fx + kC1 * alpha * g_dot_p) {
      alpha_max = alpha;
      alpha = (alpha_min + alpha_max) / 2.0;
      continue;
    } else {
      // Cache the last valid alpha value in case the curvature condition fails.
      last_valid_alpha = alpha;
    }
    Eigen::VectorXd gx_new = obj.ComputeLagrangianGradient(x_new, lambda, rho);
    // Check the curvature condition. The curvature condition checks that the
    // slope has gotten flatter, indicating progress towards the minimum.
    const double g_dot_p_new = gx_new.dot(p);
    if (std::abs(g_dot_p_new) <= kC2 * std::abs(g_dot_p)) {
      // A valid alpha value has been found.
      return alpha;
    }
    if (g_dot_p_new >= 0) {
      // For positive curvature, we need to reduce alpha.
      alpha_max = alpha;
      alpha = (alpha_min + alpha_max) / 2.0;
    } else {
      // For negative curvature, we need to increase alpha.
      alpha_min = alpha;
      alpha = alpha_max == std::numeric_limits<double>::infinity()
                  ? 2.0 * alpha
                  : (alpha_min + alpha_max) / 2.0;
    }
  }
  VLOG(3) << "Line search failed to converge after " << kMaxIters
          << " iterations. Last alpha: " << alpha << " alpha_min: " << alpha_min
          << " alpha_max: " << alpha_max
          << " last_valid_alpha: " << last_valid_alpha.value_or(-1.0);
  return last_valid_alpha.value_or(0.0);
}

}  // namespace

absl::StatusOr<Eigen::VectorXd> BfgsOptimizer::Optimize(
    const OptimizationProblem& objective) {
  Eigen::VectorXd x = objective.x0;
  Eigen::VectorXd lambda =
      Eigen::VectorXd::Ones(objective.inequality_constraints.size());
  double rho = 10.0;
  const int n = x.size();
  // Approximate inverse Hessian.
  Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd grad_fx = objective.ComputeLagrangianGradient(x, lambda, rho);

  if (n != grad_fx.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Initial input vector and objective function gradient "
                     "have different dimensions: ",
                     n, " vs ", grad_fx.size()));
  }

  constexpr int kMaxInnerIterations = 100;
  constexpr int kMaxOuterIterations = 100;
  constexpr double kToler = 1e-10;
  for (int i = 0; i < kMaxOuterIterations; ++i) {
    for (int j = 0; j < kMaxInnerIterations; ++j) {
      // Compute the search direction. Note that for the first iteration, this
      // is basically a gradient descent step.
      Eigen::VectorXd p = -B * grad_fx;
      // Compute the step size using the strong Wolfe condition line search.
      const double alpha =
          StrongWolfeConditionLineSearch(objective, x, p, lambda, rho);
      if (alpha == 0.0) {
        LOG(WARNING) << "Line search failed to find a descent direction. "
                        "Resetting Hessian approximation.";
        B = I;
        continue;
      }

      Eigen::VectorXd x_new = x + alpha * p;
      Eigen::VectorXd grad_fx_new =
          objective.ComputeLagrangianGradient(x_new, lambda, rho);
      // Check for convergence, normalized by the number of dimensions.
      if (grad_fx_new.squaredNorm() <= n * kToler * kToler ||
          (x_new - x).squaredNorm() <= n * kToler * kToler * 1e-6) {
        x = x_new;
        break;
      }

      Eigen::VectorXd s = alpha * p;
      Eigen::VectorXd y = grad_fx_new - grad_fx;

      if (const double s_dot_y = s.dot(y); s_dot_y > kToler) {
        const double rho_bfgs = 1.0 / s_dot_y;

        // auto is used here to avoid conversion to a concrete type. This
        // computes the approximate inverse Hessian.
        auto tmp_0 = I - rho_bfgs * s * y.transpose();
        auto tmp_1 = I - rho_bfgs * y * s.transpose();
        B = tmp_0 * B * tmp_1 + rho_bfgs * s * s.transpose();
      }

      x = x_new;
      grad_fx = grad_fx_new;
    }
    Eigen::VectorXd prev_lambda = lambda;
    lambda = objective.UpdateLagrangianMultipliers(x, lambda, rho);
    const double lambda_diff = (lambda - prev_lambda).squaredNorm();

    grad_fx = objective.ComputeLagrangianGradient(x, lambda, rho);

    if (lambda_diff <= lambda.size() * kToler * kToler &&
        grad_fx.squaredNorm() <= n * kToler * kToler) {
      break;
    }
    // If the constraint violation is too large, increase the penalty parameter
    // to enforce the constraints more strongly.
    if (lambda_diff > 1e-2) {
      rho = std::min(rho * 10.0, 1e6);
      // Reset the Hessian approximation if rho changed, as the Lagrangian
      // surface has changed significantly.
      B = I;
    }
  }
  return x;
}

}  // namespace grain::autotune
