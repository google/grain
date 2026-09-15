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

#include "grain/_src/python/experimental/autotune/async_fixed_ratio_node.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"
#include "grain/_src/python/experimental/autotune/softplus.h"

// We are using M/M/1/k queueing model to predict the output time, where k is
// buffer size.
// See https://en.wikipedia.org/wiki/M/M/1/K_queue for more details. Scaling
// with concurrency (n) is assumed ideal here but in b/479538575 we will explore
// using the Universal Scaling Law (USL)
// (https://www.perfdynamics.com/Manifesto/USLscalability.html) to model the
// effects of contention and coherency.

namespace grain::autotune {
namespace {
constexpr double kEpsilon = 1e-6;
constexpr double kMinUslPenalty = 1e-9;

struct UslBaseLatencies {
  double base_latency;
  double base_latency_alpha;
  double base_latency_beta;
};

// Aggregates base latency, contention (alpha), and coherency (beta) from all
// input nodes. This abstracts the latency accumulation logic needed to apply
// the Universal Scaling Law (USL) penalties further down the pipeline.
UslBaseLatencies CalculateUslBaseLatencies(
    const std::vector<std::shared_ptr<AutotuneNode>>& inputs) {
  UslBaseLatencies result = {0.0, 0.0, 0.0};
  for (const auto& input : inputs) {
    if (input == nullptr) continue;
    // We add up the input latencies (1 / throughput) to get the total latency.
    double input_latency = (input->GetBaseThroughput() > 0.0)
                               ? 1.0 / input->GetBaseThroughput()
                               : input->GetOutputTimeMs();
    result.base_latency += input_latency;
    result.base_latency_alpha += input_latency * input->GetContention();
    result.base_latency_beta += input_latency * input->GetCoherency();
  }
  return result;
}

}  // namespace

AsyncFixedRatioNode::AsyncFixedRatioNode(
    double input_ratio, std::string name, double forgetting_factor,
    std::variant<double, std::shared_ptr<AutotuneParameter>> concurrency,
    std::variant<double, std::shared_ptr<AutotuneParameter>> buffer_size)
    : AutotuneNode(name, forgetting_factor),
      input_ratio_(input_ratio),
      n_(concurrency),
      k_(buffer_size) {}

std::shared_ptr<AutotuneNode> AsyncFixedRatioNode::Copy() const {
  absl::MutexLock lock(&timing_stats_mutex_);
  auto copy = std::make_shared<AsyncFixedRatioNode>(
      input_ratio_, name_,
      timing_stats_.mean_latency_estimator.GetForgettingFactor(), n_, k_);
  copy->SetBufferRegularization(buffer_reg_mode_, buffer_reg_weight_);
  copy->SetConcurrencyRegularization(concurrency_reg_mode_,
                                     concurrency_reg_weight_);
  return copy;
}

AutotuneNodeSnapshot AsyncFixedRatioNode::GetSnapshotProto() const {
  AutotuneNodeSnapshot snapshot = AutotuneNode::GetSnapshotProto();
  auto* async_node = snapshot.mutable_async_fixed_ratio_node();
  async_node->set_input_ratio(input_ratio_);
  async_node->set_concurrency(GetConcurrency());
  async_node->set_buffer_size(GetBufferSize());
  return snapshot;
}

double AsyncFixedRatioNode::TrafficIntensity() const {
  // Arrival rate / service rate.
  const double producer_time_ms = GetProducerTimeMs();
  return GetConsumerTimeMs() / std::max(kMinUslPenalty, producer_time_ms);
}

double AsyncFixedRatioNode::GetProducerTimeMsIdeal() const {
  const double n = std::max(1.0, GetConcurrency());

  UslBaseLatencies latencies = CalculateUslBaseLatencies(inputs_);

  if (latencies.base_latency > kMinUslPenalty) {
    double alpha = latencies.base_latency_alpha / latencies.base_latency;
    double beta = latencies.base_latency_beta / latencies.base_latency;
    double usl_penalty = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0);
    // Apply Softplus to smoothly clamp production time at 0.
    return input_ratio_ * latencies.base_latency * Softplus(usl_penalty) / n;
  }

  return 0.0;
}

double AsyncFixedRatioNode::GetProducerTimeMs() const {
  const double pt_ideal = GetProducerTimeMsIdeal();
  if (pt_ideal <= 0.0) return pt_ideal;

  const double n = std::max(1.0, GetConcurrency());
  const double k = std::max(1.0, GetBufferSize());
  const double deficit_ratio = (2.0 * n - k) / k;
  const double buffer_penalty =
      1.0 + kBufferContentionFactor * Softplus(deficit_ratio);

  return pt_ideal * buffer_penalty;
}

double AsyncFixedRatioNode::GetOutputTimeMs() const {
  const double pt = GetProducerTimeMs();
  const double rho = TrafficIntensity();
  const double mu_inv = std::max(kMinUslPenalty, GetConsumerTimeMs());
  const double k = GetBufferSize();
  const double n = std::max(1.0, GetConcurrency());

  double reg_penalty = 0.0;
  if (buffer_reg_mode_ == BufferRegularizationMode::kL2 &&
      buffer_reg_weight_ > 0.0) {
    reg_penalty = buffer_reg_weight_ * (k / 256.0) * (k / 256.0) * mu_inv;
  } else if (buffer_reg_mode_ == BufferRegularizationMode::kBarrier &&
             buffer_reg_weight_ > 0.0) {
    constexpr double kTargetBufferMultiplier = 3.0;
    const double u = k / (kTargetBufferMultiplier * n);
    constexpr double kEpsBarrier = 1e-4;
    const double safe_u = std::max(kEpsBarrier, u);
    reg_penalty = buffer_reg_weight_ * mu_inv * (u - std::log(safe_u) - 1.0);
  }

  if (concurrency_reg_mode_ == ConcurrencyRegularizationMode::kLinear &&
      concurrency_reg_weight_ > 0.0) {
    const double n_max = GetMaxConcurrency();
    reg_penalty += concurrency_reg_weight_ * mu_inv * (n / n_max);
  } else if (concurrency_reg_mode_ ==
                 ConcurrencyRegularizationMode::kQuadratic &&
             concurrency_reg_weight_ > 0.0) {
    const double n_max = GetMaxConcurrency();
    reg_penalty += concurrency_reg_weight_ * mu_inv * (n / n_max) * (n / n_max);
  }

  if (rho < kEpsilon) return pt + reg_penalty;

  // Mathematical limit: patch the 0/0 hole
  if (std::abs(rho - 1.0) < kEpsilon) {
    return mu_inv * (1.0 + 1.0 / (k + 1.0)) + reg_penalty;
  }

  // We compute L = (k + 1) * ln(rho) and branch on its sign to ensure numerical
  // stability. If rho > 1 (L > 0), computing rho^(k+1) directly would overflow
  // to infinity. By scaling the numerator and denominator by e^{-L}, we
  // elegantly drive the fraction to 0 as rho approaches infinity, without
  // arbitrary limits.
  double L = (k + 1.0) * std::log(rho);

  if (L <= 0.0) {
    // pt substitution natively handles rho = 0 without exploding
    return mu_inv + pt * ((1.0 - rho) / -std::expm1(L)) + reg_penalty;
  } else {
    // e^{-L} scaling drives fraction to 0 as rho -> infinity
    return mu_inv + pt * ((rho - 1.0) * std::exp(-L) / -std::expm1(-L)) +
           reg_penalty;
  }
}

double AsyncFixedRatioNode::GetSelfMemoryUsage() const {
  return GetBufferSize() * GetElementSizeBytes();
}

double AsyncFixedRatioNode::GetMemoryUsage() const {
  // Memory usage M(node) = GetConcurrencyMemoryUsage(SUM_i(M(input)), n) +
  // GetSelfMemoryUsage()
  double memory_usage_inputs = 0;
  for (const auto& input : inputs_) {
    if (input == nullptr) continue;
    memory_usage_inputs += input->GetMemoryUsage();
  }
  return GetConcurrencyMemoryUsage(memory_usage_inputs, GetConcurrency()) +
         GetSelfMemoryUsage();
}

double AsyncFixedRatioNode::GetConcurrencyMemoryUsage(
    double memory_usage_inputs, double concurrency) {
  // TODO: b/493302333 - This may need to be updated with base memory usage of a
  // thread/process.
  return concurrency * memory_usage_inputs;
}

double AsyncFixedRatioNode::TrafficIntensitySensitivity() const {
  const double rho = TrafficIntensity();
  const double mu_inv = std::max(kMinUslPenalty, GetConsumerTimeMs());
  const double k = GetBufferSize();

  // Physical limit: derivative approaches -infinity
  if (rho < kEpsilon) return -mu_inv / std::max(kMinUslPenalty, rho * rho);

  // Mathematical limit: patch the 0/0 hole
  if (std::abs(rho - 1.0) < kEpsilon) {
    return -(mu_inv * (k + 2.0)) / (2.0 * (k + 1.0));
  }

  double L = (k + 1.0) * std::log(rho);
  double x = (k + 1.0) * rho - k - 2.0;

  if (L <= 0.0) {
    double exp_L = std::exp(L);
    double expm1_L = std::expm1(L);
    return -(exp_L * x + 1.0) /
           ((1.0 / mu_inv) * rho * rho * expm1_L * expm1_L);
  } else {
    // Scaled by e^{-2L} to naturally decay to 0
    double exp_minus_L = std::exp(-L);
    double expm1_minus_L = std::expm1(-L);
    return -(x * exp_minus_L + std::exp(-2.0 * L)) /
           ((1.0 / mu_inv) * rho * rho * expm1_minus_L * expm1_minus_L);
  }
}

double AsyncFixedRatioNode::BufferSizeSensitivity() const {
  const double pt_ideal = GetProducerTimeMsIdeal();
  const double k = std::max(1.0, GetBufferSize());
  const double n = std::max(1.0, GetConcurrency());
  const double deficit_ratio = (2.0 * n - k) / k;
  const double dPT_dk = -pt_ideal * kBufferContentionFactor *
                        SoftplusDerivative(deficit_ratio) * (2.0 * n / (k * k));

  const double rho = TrafficIntensity();
  const double mu_inv = std::max(kMinUslPenalty, GetConsumerTimeMs());

  // C++ Trap: bypass 0 * -Inf = NaN in the numerator
  if (rho < kEpsilon) return dPT_dk;

  // Mathematical limit: patch the 0/0 hole
  double dO_dk_direct = 0.0;
  if (std::abs(rho - 1.0) < kEpsilon) {
    dO_dk_direct = -mu_inv / ((k + 1.0) * (k + 1.0));
  } else {
    double log_rho = std::log(rho);
    double L = (k + 1.0) * log_rho;
    double y = (rho - 1.0) * log_rho;

    if (L <= 0.0) {
      dO_dk_direct = -(std::exp(k * log_rho) * y) /
                     ((1.0 / mu_inv) * std::pow(std::expm1(L), 2));
    } else {
      // Scaled by e^{-2L} to naturally decay to 0
      dO_dk_direct = -(std::exp(-(k + 2.0) * log_rho) * y) /
                     ((1.0 / mu_inv) * std::pow(std::expm1(-L), 2));
    }
  }

  const double pt_buffered = GetProducerTimeMs();
  const double dO_drho = TrafficIntensitySensitivity();
  const double dO_dpt =
      pt_buffered > kMinUslPenalty ? -rho * dO_drho / pt_buffered : 1.0;

  double d_reg_dk = 0.0;
  if (buffer_reg_mode_ == BufferRegularizationMode::kL2 &&
      buffer_reg_weight_ > 0.0) {
    d_reg_dk = 2.0 * buffer_reg_weight_ * (k / (256.0 * 256.0)) * mu_inv;
  } else if (buffer_reg_mode_ == BufferRegularizationMode::kBarrier &&
             buffer_reg_weight_ > 0.0) {
    constexpr double kTargetBufferMultiplier = 3.0;
    d_reg_dk = buffer_reg_weight_ * mu_inv *
               (1.0 / (kTargetBufferMultiplier * n) - 1.0 / std::max(1e-4, k));
  }

  return dO_dk_direct + dO_dpt * dPT_dk + d_reg_dk;
}

double AsyncFixedRatioNode::TrafficIntensityWrtConcurrencySensitivity() const {
  // Sensitivity of traffic intensity rho with respect to concurrency n.
  const double mu_inv = std::max(kMinUslPenalty, GetConsumerTimeMs());
  const double pt = GetProducerTimeMs();

  if (pt < kMinUslPenalty) {
    return 0.0;
  }

  const double dPT_dn = GetProducerTimeMsWrtConcurrencySensitivity();
  return -mu_inv * dPT_dn / (pt * pt);
}

double AsyncFixedRatioNode::GetProducerTimeMsWrtConcurrencySensitivity() const {
  const double n = std::max(1.0, GetConcurrency());
  const double k = std::max(1.0, GetBufferSize());
  const double deficit_ratio = (2.0 * n - k) / k;
  const double buffer_penalty =
      1.0 + kBufferContentionFactor * Softplus(deficit_ratio);

  UslBaseLatencies latencies = CalculateUslBaseLatencies(inputs_);
  if (latencies.base_latency <= kMinUslPenalty) return 0.0;

  double alpha = latencies.base_latency_alpha / latencies.base_latency;
  double beta = latencies.base_latency_beta / latencies.base_latency;
  const double usl_penalty_raw = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0);
  const double usl_penalty_safe = Softplus(usl_penalty_raw);
  const double usl_penalty_derivative_safe =
      SoftplusDerivative(usl_penalty_raw) * (alpha + beta * (2.0 * n - 1.0));
  const double dPT_ideal_dn =
      input_ratio_ * latencies.base_latency *
      (n * usl_penalty_derivative_safe - usl_penalty_safe) / (n * n);

  const double pt_ideal = GetProducerTimeMsIdeal();
  const double dPenalty_dn =
      kBufferContentionFactor * SoftplusDerivative(deficit_ratio) * (2.0 / k);

  return dPT_ideal_dn * buffer_penalty + pt_ideal * dPenalty_dn;
}

std::vector<double> AsyncFixedRatioNode::ArrivalRateSensitivityWrtInputTime()
    const {
  // Sensitivity of arrival rate lambda with respect to input time.
  const double pt = GetProducerTimeMs();
  const double n = std::max(1.0, GetConcurrency());

  if (pt < kMinUslPenalty || GetBaseThroughput() > 0.0) {
    return std::vector<double>(inputs_.size(), 0.0);
  }

  UslBaseLatencies latencies = CalculateUslBaseLatencies(inputs_);

  if (latencies.base_latency < kMinUslPenalty)
    return std::vector<double>(inputs_.size(), 0.0);

  double alpha = latencies.base_latency_alpha / latencies.base_latency;
  double beta = latencies.base_latency_beta / latencies.base_latency;

  // Apply Softplus smoothing to avoid clamping at zero and losing gradients
  // during backprop when the optimizer hits negative latency predictions.
  const double usl_penalty_raw = 1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0);
  const double usl_penalty_safe = Softplus(usl_penalty_raw);

  const double pt_ideal = std::max(kMinUslPenalty, GetProducerTimeMsIdeal());
  const double buffer_penalty = pt / pt_ideal;

  return std::vector<double>(inputs_.size(),
                             -input_ratio_ * usl_penalty_safe * buffer_penalty /
                                 std::max(kMinUslPenalty, n * pt * pt));
}

double AsyncFixedRatioNode::GetMaxConcurrency() const {
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(n_)) {
    const auto& param = std::get<std::shared_ptr<AutotuneParameter>>(n_);
    if (param != nullptr) {
      double max_val = param->GetRange().second;
      if (std::isfinite(max_val) && max_val > 0.0 &&
          max_val < std::numeric_limits<double>::max() / 2.0) {
        return max_val;
      }
    }
  }
  return 64.0;
}

double AsyncFixedRatioNode::GetConcurrency() const {
  if (std::holds_alternative<double>(n_)) {
    return std::get<double>(n_);
  }
  return std::get<std::shared_ptr<AutotuneParameter>>(n_)->GetValue();
}

double AsyncFixedRatioNode::GetBufferSize() const {
  if (std::holds_alternative<double>(k_)) {
    return std::get<double>(k_);
  }
  return std::get<std::shared_ptr<AutotuneParameter>>(k_)->GetValue();
}

AutotuneNode::PartialDerivatives
AsyncFixedRatioNode::ComputePartialDerivatives() const {
  const double dO_drho = TrafficIntensitySensitivity();
  const double rho = TrafficIntensity();

  absl::flat_hash_map<int64_t, double> parameter_gradients;

  const double mu_inv = std::max(kMinUslPenalty, GetConsumerTimeMs());
  const double n = std::max(1.0, GetConcurrency());
  const double k = std::max(1.0, GetBufferSize());

  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(k_)) {
    parameter_gradients[std::get<std::shared_ptr<AutotuneParameter>>(k_)
                            ->id()] = BufferSizeSensitivity();
  }
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(n_)) {
    double dO_dn = rho < kEpsilon
                       ? GetProducerTimeMsWrtConcurrencySensitivity()
                       : dO_drho * TrafficIntensityWrtConcurrencySensitivity();
    double d_reg_dn = 0.0;
    if (buffer_reg_mode_ == BufferRegularizationMode::kBarrier &&
        buffer_reg_weight_ > 0.0) {
      constexpr double kTargetBufferMultiplier = 3.0;
      d_reg_dn +=
          buffer_reg_weight_ * mu_inv *
          (-k / (kTargetBufferMultiplier * n * n) + 1.0 / std::max(1e-4, n));
    }
    if (concurrency_reg_mode_ == ConcurrencyRegularizationMode::kLinear &&
        concurrency_reg_weight_ > 0.0) {
      const double n_max = GetMaxConcurrency();
      d_reg_dn += concurrency_reg_weight_ * mu_inv * (1.0 / n_max);
    } else if (concurrency_reg_mode_ ==
                   ConcurrencyRegularizationMode::kQuadratic &&
               concurrency_reg_weight_ > 0.0) {
      const double n_max = GetMaxConcurrency();
      d_reg_dn +=
          2.0 * concurrency_reg_weight_ * mu_inv * (n / (n_max * n_max));
    }
    parameter_gradients[std::get<std::shared_ptr<AutotuneParameter>>(n_)
                            ->id()] = dO_dn + d_reg_dn;
  }

  std::vector<double> tmp_grads = ArrivalRateSensitivityWrtInputTime();
  for (auto& grad : tmp_grads) {
    grad *= mu_inv * dO_drho;
  }

  return PartialDerivatives{
      .wrt_inputs = std::move(tmp_grads),
      .wrt_params = std::move(parameter_gradients),
  };
}

AutotuneNode::PartialDerivatives
AsyncFixedRatioNode::ComputeMemoryUsagePartialDerivatives() const {
  absl::flat_hash_map<int64_t, double> parameter_gradients;
  // Partial derivative of memory usage w.r.t buffer size.
  double dM_dk = GetElementSizeBytes();
  // Partial derivative of memory usage w.r.t concurrency.
  double dM_dn = 0;
  for (const auto& input : inputs_) {
    if (input == nullptr) continue;
    dM_dn += input->GetMemoryUsage();
  }
  // Partial derivative of memory usage w.r.t input memory usage.
  double dM_dM_input = GetConcurrency();

  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(k_)) {
    parameter_gradients[std::get<std::shared_ptr<AutotuneParameter>>(k_)
                            ->id()] = dM_dk;
  }
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(n_)) {
    parameter_gradients[std::get<std::shared_ptr<AutotuneParameter>>(n_)
                            ->id()] = dM_dn;
  }
  return PartialDerivatives{
      .wrt_inputs = std::vector<double>(inputs_.size(), dM_dM_input),
      .wrt_params = std::move(parameter_gradients),
  };
}

void AsyncFixedRatioNode::GetLocalTunableParameters(
    std::vector<std::shared_ptr<AutotuneParameter>>& parameters) const {
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(n_)) {
    parameters.push_back(std::get<std::shared_ptr<AutotuneParameter>>(n_));
  }
  if (std::holds_alternative<std::shared_ptr<AutotuneParameter>>(k_)) {
    parameters.push_back(std::get<std::shared_ptr<AutotuneParameter>>(k_));
  }
}

}  // namespace grain::autotune
