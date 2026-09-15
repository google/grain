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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_RECURSIVE_LEAST_SQUARES_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_RECURSIVE_LEAST_SQUARES_H_

#include <algorithm>
#include <type_traits>

#include "Eigen/Core"
#include "Eigen/LU"  // IWYU pragma: keep
#include "absl/status/status.h"

namespace grain::autotune {

// Recursive Least Squares (RLS) filter for estimating coefficients of a linear
// model with Subspace of Information Forgetting (SIFt-RLS).
// This class maintains the current estimates of the coefficients, the
// covariance matrix, and the information matrix to prevent estimator windup.
// This class is not thread-safe.
template <int NumParameters>
class RecursiveLeastSquares {
 public:
  using VectorType = Eigen::Matrix<double, NumParameters, 1>;
  using MatrixType = Eigen::Matrix<double, NumParameters, NumParameters>;

  static constexpr double kMinForgettingFactor = 1e-4;

  // Factory to initialize with specific estimate.
  // Enabled only for NumParameters == 1.
  template <int N = NumParameters,
            typename std::enable_if<(N == 1), int>::type = 0>
  static RecursiveLeastSquares WithEstimate(double initial_estimate,
                                            double forgetting_factor = 0.99,
                                            double initial_variance = 1e5) {
    return RecursiveLeastSquares(VectorType::Constant(initial_estimate),
                                 forgetting_factor, initial_variance);
  }

  // Factory to initialize with a fully specified internal state.
  static RecursiveLeastSquares WithState(const VectorType& estimates,
                                         const MatrixType& covariance,
                                         double forgetting_factor) {
    RecursiveLeastSquares rls(forgetting_factor);
    rls.estimates_ = estimates;
    rls.p_ = covariance;
    rls.r_ = covariance.inverse();
    return rls;
  }

  // Initialize with zero estimates.
  // initial_variance is the diagonal value for the initial covariance matrix.
  RecursiveLeastSquares(double forgetting_factor = 0.99,
                        double initial_variance = 1e6)
      : forgetting_factor_(std::max(kMinForgettingFactor, forgetting_factor)),
        estimates_(VectorType::Zero()),
        p_(MatrixType::Identity() * initial_variance),
        r_(MatrixType::Identity() * (1.0 / initial_variance)),
        prior_estimates_(VectorType::Zero()),
        prior_variance_(VectorType::Constant(initial_variance)) {}

  // Initialize with initial coefficient estimates.
  // initial_variance is the diagonal value for the initial covariance matrix.
  RecursiveLeastSquares(const VectorType& initial_estimates,
                        double forgetting_factor = 0.99,
                        double initial_variance = 1e6)
      : forgetting_factor_(std::max(kMinForgettingFactor, forgetting_factor)),
        estimates_(initial_estimates),
        p_(MatrixType::Identity() * initial_variance),
        r_(MatrixType::Identity() * (1.0 / initial_variance)),
        prior_estimates_(initial_estimates),
        prior_variance_(VectorType::Constant(initial_variance)) {}

  // Update the estimator with a new observation consisting of an input vector
  // and an observation.
  absl::Status Update(const VectorType& input, double observation) {
    return UpdateInternal(input, observation);
  }

  // Updates the coefficient estimates and covariance matrix using a scalar
  // input and observation.
  // Enabled only for NumParameters == 1.
  template <int N = NumParameters,
            typename std::enable_if<(N == 1), int>::type = 0>
  absl::Status Update(double input, double observation) {
    VectorType input_vector;
    input_vector[0] = input;
    return UpdateInternal(input_vector, observation);
  }

  // Updates the coefficient estimates and covariance matrix using a new
  // observation, assuming input is 1.0.
  // Enabled only for NumParameters == 1.
  template <int N = NumParameters,
            typename std::enable_if<(N == 1), int>::type = 0>
  absl::Status Update(double observation) {
    return Update(1.0, observation);
  }

  const VectorType& GetEstimates() const { return estimates_; }
  MatrixType GetCovariance() const { return p_; }
  MatrixType GetInformationMatrix() const { return r_; }

  // Scalar accessors enabled only for NumParameters == 1.
  template <int N = NumParameters,
            typename std::enable_if<(N == 1), int>::type = 0>
  double GetEstimate() const {
    return estimates_[0];
  }

  template <int N = NumParameters,
            typename std::enable_if<(N == 1), int>::type = 0>
  double GetVariance() const {
    return p_(0, 0);
  }

  double GetForgettingFactor() const { return forgetting_factor_; }

 private:
  // Internal update logic utilizing Subspace of Information Forgetting.
  // Reference: "SIFt-RLS: Subspace of Information Forgetting Recursive Least
  // Squares" arXiv: https://arxiv.org/abs/2404.10844
  absl::Status UpdateInternal(const VectorType& input, double observation) {
    // Threshold to evaluate if the incoming direction has persistent excitation
    constexpr double kEpsilon = 1e-8;
    // Threshold to protect against division by zero.
    constexpr double kDivisionByZeroThreshold = 1e-12;

    // Information filtering
    // The SIFt-RLS paper specifies truncating singular values below
    // sqrt(epsilon). If the input is below this threshold, it is treated as 0,
    // meaning no update occurs.
    if (input.squaredNorm() < kEpsilon) {
      return absl::OkStatus();
    }

    MatrixType p_bar = p_;
    MatrixType r_bar = r_;

    // SIFting (directional forgetting)
    // Calculate scalar projection against the information matrix
    double c = input.dot(r_ * input);

    if (c > kDivisionByZeroThreshold) {  // Protect against division by zero
      // Modify covariance matrix purely in the direction of the input
      p_bar = p_ + ((1.0 - forgetting_factor_) / (forgetting_factor_ * c)) *
                       (input * input.transpose());

      // Update the information matrix using the SIFt identity
      // Avoids an O(N^3) dense matrix multiplication compared to original form
      VectorType r_input = r_ * input;
      r_bar = r_ - ((1.0 - forgetting_factor_) / c) *
                       (r_input * r_input.transpose());

      // Enforce symmetry to prevent floating-point drift
      r_bar = (0.5 * (r_bar + r_bar.transpose())).eval();
    }

    // Standard RLS update using the modified (SIFted) matrices
    double prediction = input.dot(estimates_);
    double error = observation - prediction;

    VectorType p_bar_input = p_bar * input;
    double denominator = 1.0 + input.dot(p_bar_input);
    VectorType gain = p_bar_input / denominator;

    // Final recursive updates for parameter estimates, covariance, and
    // information matrices
    VectorType new_estimates = estimates_ + gain * error;
    MatrixType new_p = p_bar - gain * p_bar_input.transpose();
    MatrixType new_r = r_bar + input * input.transpose();

    // Check for NaN or Inf to prevent estimator collapse.
    if (!new_estimates.allFinite() || !new_p.allFinite() ||
        !new_r.allFinite()) {
      return absl::InternalError("RLS update resulted in non-finite values.");
    }

    estimates_ = new_estimates;
    p_ = (0.5 * (new_p + new_p.transpose())).eval();
    r_ = (0.5 * (new_r + new_r.transpose())).eval();

    return absl::OkStatus();
  }

  double forgetting_factor_;
  VectorType estimates_;

  // Track both Covariance (p_) and Information (r_) matrices for directional
  // updates
  MatrixType p_;
  MatrixType r_;

  VectorType prior_estimates_;
  VectorType prior_variance_;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_RECURSIVE_LEAST_SQUARES_H_
