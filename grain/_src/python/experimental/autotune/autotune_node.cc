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

#include "grain/_src/python/experimental/autotune/autotune_node.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"
#include "grain/_src/python/experimental/autotune/recursive_least_squares.h"
#include "grain/_src/python/experimental/autotune/softplus.h"

namespace grain::autotune {
namespace {

// The minimum value for the USL penalty term to avoid division by zero or
// negative throughput.
constexpr double kMinUslPenalty = 1e-9;

template <int NumParameters>
AutotuneNodeSnapshot::RlsStats ToRlsStatsProto(
    const RecursiveLeastSquares<NumParameters>& rls) {
  AutotuneNodeSnapshot::RlsStats stats;
  stats.set_forgetting_factor(rls.GetForgettingFactor());
  for (int i = 0; i < NumParameters; ++i) {
    stats.add_estimates(rls.GetEstimates()(i));
  }
  for (int i = 0; i < NumParameters; ++i) {
    for (int j = 0; j < NumParameters; ++j) {
      stats.add_covariance(rls.GetCovariance()(i, j));
    }
  }
  return stats;
}

absl::Status ValidateAutotuneGraph(
    const std::shared_ptr<const AutotuneNode>& node) {
  absl::flat_hash_set<const AutotuneNode*> visited_nodes;
  std::vector<std::shared_ptr<const AutotuneNode>> stack;
  stack.push_back(node);

  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    if (!visited_nodes.insert(node.get()).second) {
      return absl::InvalidArgumentError("Cycle detected in node graph.");
    }
    for (const auto& input : node->GetInputs()) {
      stack.push_back(input);
    }
  }

  return absl::OkStatus();
}

}  // namespace

AutotuneNode::Timer::Timer() : id_(next_id_++) {}

absl::StatusOr<absl::Duration> AutotuneNode::Timer::Start() {
  if (GetState().state != TimerState::kInactive) {
    return absl::FailedPreconditionError("Timer is in the wrong state.");
  }
  auto& state = GetState();
  state.time_start = absl::Now();
  state.time_elapsed = absl::ZeroDuration();
  state.num_pauses = 0;
  state.state = TimerState::kActive;
  if (state.time_end != absl::InfinitePast()) {
    return state.time_start - state.time_end;
  }
  return absl::ZeroDuration();
}

absl::StatusOr<absl::Duration> AutotuneNode::Timer::End() {
  if (GetState().state == TimerState::kInactive) {
    return absl::FailedPreconditionError(
        absl::StrCat("Timer ", id_, " is already inactive."));
  }
  auto& state = GetState();
  if (state.state == TimerState::kActive) {
    state.time_elapsed += absl::Now() - state.time_start;
  }
  state.state = TimerState::kInactive;
  state.time_end = absl::Now();
  return state.time_elapsed;
}

absl::Status AutotuneNode::Timer::Pause() {
  if (GetState().state != TimerState::kActive) {
    return absl::FailedPreconditionError(
        absl::StrCat("Timer ", id_, " cannot be paused from state ",
                     static_cast<int>(GetState().state)));
  }
  auto& state = GetState();
  state.time_elapsed += absl::Now() - state.time_start;
  state.num_pauses++;
  state.state = TimerState::kPaused;
  return absl::OkStatus();
}

absl::Status AutotuneNode::Timer::Resume() {
  if (GetState().state != TimerState::kPaused) {
    return absl::FailedPreconditionError(
        absl::StrCat("Timer ", id_, " cannot be resumed from state ",
                     static_cast<int>(GetState().state)));
  }
  auto& state = GetState();
  state.time_start = absl::Now();
  state.state = TimerState::kActive;
  return absl::OkStatus();
}

int64_t AutotuneNode::Timer::NumPauses() { return GetState().num_pauses; }

AutotuneNode::Timer::TimerState& AutotuneNode::Timer::GetState() {
  registry_.resize(std::max<size_t>(registry_.size(), id_ + 1));
  return registry_[id_];
}

AutotuneNode::AutotuneNode(std::string name, double forgetting_factor)
    : name_(name) {
  timing_stats_.mean_latency_estimator =
      RecursiveLeastSquares<1>(forgetting_factor);
  timing_stats_.mean_inactive_time_estimator =
      RecursiveLeastSquares<1>(forgetting_factor);
  throughput_stats_.mean_element_size_bytes_estimator =
      RecursiveLeastSquares<1>(forgetting_factor);

  Eigen::Matrix<double, 3, 3> initial_cov =
      Eigen::Matrix<double, 3, 3>::Identity() * 1e4;
  initial_cov(0, 0) = 1.0;  // Regularization for normalized c0 (beta)
  concurrency_model_.usl_estimator = RecursiveLeastSquares<3>::WithState(
      Eigen::Matrix<double, 3, 1>::Zero(), initial_cov, 0.999);
}

absl::Status AutotuneNode::Validate() const {
  return ValidateAutotuneGraph(shared_from_this());
}

absl::StatusOr<std::shared_ptr<AutotuneNode>> AutotuneNode::AddInput(
    std::shared_ptr<AutotuneNode> input) {
  auto self_reference = shared_from_this();
  if (input->output_.lock() != nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("Node ", input->GetName(), " already has an output."));
  }
  input->output_ = self_reference;
  inputs_.push_back(input);
  return self_reference;
}

std::shared_ptr<AutotuneNode> AutotuneNode::GetSnapshot() const {
  std::shared_ptr<AutotuneNode> snapshot = Copy();
  CopyStatsTo(snapshot.get());
  snapshot->inputs_.clear();

  for (auto& input : inputs_) {
    auto input_snapshot = input->GetSnapshot();
    input_snapshot->output_ = snapshot;
    snapshot->inputs_.push_back(input_snapshot);
  }
  return snapshot;
}

void AutotuneNode::CopyStatsTo(AutotuneNode* dest) const {
  TimingStats temp_timing_stats;
  {
    absl::MutexLock src_lock(&timing_stats_mutex_);
    temp_timing_stats = timing_stats_;
  }
  {
    absl::MutexLock dest_lock(&dest->timing_stats_mutex_);
    dest->timing_stats_ = temp_timing_stats;
  }

  ThroughputStats temp_throughput_stats;
  {
    absl::MutexLock src_lock(&throughput_stats_mutex_);
    temp_throughput_stats = throughput_stats_;
  }
  {
    absl::MutexLock dest_lock(&dest->throughput_stats_mutex_);
    dest->throughput_stats_ = temp_throughput_stats;
  }

  ConcurrencyModel temp_concurrency_model;
  {
    absl::MutexLock src_lock(&concurrency_model_mutex_);
    const_cast<AutotuneNode*>(this)->concurrency_model_.FlushPendingUpdates();
    temp_concurrency_model = concurrency_model_;
  }
  {
    absl::MutexLock dest_lock(&dest->concurrency_model_mutex_);
    dest->concurrency_model_ = temp_concurrency_model;
  }
}

AutotuneNodeSnapshot AutotuneNode::GetSnapshotProto() const {
  AutotuneNodeSnapshot snapshot;
  snapshot.set_name(name_);
  {
    absl::MutexLock lock(&timing_stats_mutex_);
    snapshot.set_forgetting_factor(
        timing_stats_.mean_latency_estimator.GetForgettingFactor());
    auto* timing = snapshot.mutable_timing_stats();
    *timing->mutable_mean_latency() =
        ToRlsStatsProto(timing_stats_.mean_latency_estimator);
    *timing->mutable_mean_inactive_time() =
        ToRlsStatsProto(timing_stats_.mean_inactive_time_estimator);
    timing->set_count(timing_stats_.count);
  }
  {
    absl::MutexLock lock(&throughput_stats_mutex_);
    auto* throughput = snapshot.mutable_throughput_stats();
    *throughput->mutable_mean_element_size_bytes() =
        ToRlsStatsProto(throughput_stats_.mean_element_size_bytes_estimator);
    throughput->set_min_element_size_bytes(
        throughput_stats_.min_element_size_bytes);
    throughput->set_max_element_size_bytes(
        throughput_stats_.max_element_size_bytes);
  }
  {
    absl::MutexLock lock(&concurrency_model_mutex_);
    const_cast<AutotuneNode*>(this)->concurrency_model_.FlushPendingUpdates();
    auto* concurrency = snapshot.mutable_concurrency_stats();
    *concurrency->mutable_usl_estimator() =
        ToRlsStatsProto(concurrency_model_.usl_estimator);
    concurrency->set_num_observations(concurrency_model_.num_observations_);
    concurrency->set_concurrency_bins_seen(
        concurrency_model_.concurrency_bins_seen.to_ullong());
  }
  for (const auto& input : inputs_) {
    *snapshot.add_inputs() = input->GetSnapshotProto();
  }
  return snapshot;
}

absl::StatusOr<absl::Duration> AutotuneNode::RecordStart() {
  auto inactive_duration_or = timer_.Start();
  if (!inactive_duration_or.ok()) {
    return inactive_duration_or.status();
  }
  absl::Duration inactive_duration = inactive_duration_or.value();
  if (inactive_duration > absl::ZeroDuration()) {
    absl::MutexLock lock(&timing_stats_mutex_);
    auto status = timing_stats_.mean_inactive_time_estimator.Update(
        absl::ToDoubleMilliseconds(inactive_duration));
    if (!status.ok()) {
      return status;
    }
  }
  absl::MutexLock lock(&concurrency_model_mutex_);
  // Note: the concurrency timer is never paused. It is inclusive of input
  // times.

  auto now = absl::Now();
  if (concurrency_model_.last_integral_time == absl::InfinitePast()) {
    concurrency_model_.last_integral_time = now;
  }
  concurrency_model_.integral_active_threads_ms +=
      concurrency_model_.active_threads *
      absl::ToDoubleMilliseconds(now - concurrency_model_.last_integral_time);
  concurrency_model_.last_integral_time = now;

  concurrency_model_.active_threads++;
  concurrency_model_.timer.SetStartIntegral(
      concurrency_model_.integral_active_threads_ms);

  auto start_or = concurrency_model_.timer.Start();
  if (!start_or.ok()) {
    return start_or.status();
  }
  return inactive_duration;
}

absl::Status AutotuneNode::RecordPause() {
  auto status = timer_.Pause();
  if (!status.ok()) {
    VLOG(1) << "Swallowing RecordPause error on preemption/teardown: "
            << status;
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status AutotuneNode::RecordResume() {
  auto status = timer_.Resume();
  if (!status.ok()) {
    VLOG(1) << "Swallowing RecordResume error on preemption/teardown: "
            << status;
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status AutotuneNode::RecordElementSizeBytes(int64_t bytes) {
  absl::MutexLock lock(&throughput_stats_mutex_);
  if (throughput_stats_.min_element_size_bytes == -1 ||
      bytes < throughput_stats_.min_element_size_bytes) {
    throughput_stats_.min_element_size_bytes = bytes;
  }
  if (bytes > throughput_stats_.max_element_size_bytes) {
    throughput_stats_.max_element_size_bytes = bytes;
  }
  return throughput_stats_.mean_element_size_bytes_estimator.Update(
      static_cast<double>(bytes));
}

absl::StatusOr<absl::Duration> AutotuneNode::RecordEnd() {
  auto elapsed_or_status = timer_.End();
  if (!elapsed_or_status.ok()) {
    VLOG(1) << "Swallowing RecordEnd error on inactive timer during preemption "
               "or teardown: "
            << elapsed_or_status.status();
    return absl::ZeroDuration();
  }
  absl::Duration time_elapsed = elapsed_or_status.value();
  {
    absl::MutexLock lock(&timing_stats_mutex_);
    timing_stats_.count++;
    auto status = timing_stats_.mean_latency_estimator.Update(
        absl::ToDoubleMilliseconds(time_elapsed));
    if (!status.ok()) {
      return status;
    }
  }
  {
    absl::MutexLock lock(&concurrency_model_mutex_);
    auto now = absl::Now();
    if (concurrency_model_.last_integral_time == absl::InfinitePast()) {
      concurrency_model_.last_integral_time = now;
    }
    concurrency_model_.integral_active_threads_ms +=
        concurrency_model_.active_threads *
        absl::ToDoubleMilliseconds(now - concurrency_model_.last_integral_time);
    concurrency_model_.last_integral_time = now;

    concurrency_model_.active_threads--;

    auto end_or = concurrency_model_.timer.End();
    if (!end_or.ok()) {
      return end_or.status();
    }
    time_elapsed = end_or.value();

    double elapsed_ms = absl::ToDoubleMilliseconds(time_elapsed);
    double start_integral = concurrency_model_.timer.GetStartIntegral();

    double n = 1.0;
    if (elapsed_ms > 0) {
      n = (concurrency_model_.integral_active_threads_ms - start_integral) /
          elapsed_ms;
      n = std::max<double>(1.0, n);
    }

    double obs = elapsed_ms / n;
    concurrency_model_.history.push_back({n, obs});
    if (concurrency_model_.history.size() > 1000) {
      concurrency_model_.history.pop_front();
    }
    concurrency_model_.num_observations_++;
    double raw_x0 = n - 1.0;
    double raw_x1 = (n - 1.0) / n;
    double raw_x2 = 1.0 / n;
    Eigen::Vector3d norm_x(raw_x0 / ConcurrencyModel::kScaleX0,
                           raw_x1 / ConcurrencyModel::kScaleX1,
                           raw_x2 / ConcurrencyModel::kScaleX2);
    concurrency_model_.pending_rls_updates.push_back({norm_x, obs});
    if (concurrency_model_.pending_rls_updates.size() >=
        ConcurrencyModel::kRlsBatchSize) {
      concurrency_model_.FlushPendingUpdates();
      bool previously_active = concurrency_model_.spectral_activation_logged;
      if (!previously_active &&
          concurrency_model_.ComputeSpectralActivationWeight() >= 0.5) {
        concurrency_model_.spectral_activation_logged = true;
        LOG(INFO)
            << "AutotuneNode (" << name_
            << "): Fisher Information Matrix achieved spectral conditioning."
            << " Activating USL contention (alpha) and coherency (beta) model.";
      }
    }
  }
  return time_elapsed;
}

double AutotuneNode::GetSelfTimeMs() const {
  absl::MutexLock lock(&timing_stats_mutex_);
  return timing_stats_.mean_latency_estimator.GetEstimate();
}

double AutotuneNode::GetSelfMemoryUsage() const { return 0; }

double AutotuneNode::GetOutputTimeMs() const {
  double self_time_ms = GetSelfTimeMs();
  double input_time_ms = 0.;
  for (const auto& input : inputs_) {
    input_time_ms += input->GetOutputTimeMs();
  }
  return self_time_ms + input_time_ms;
}

double AutotuneNode::GetMemoryUsage() const {
  double self_memory_usage = GetSelfMemoryUsage();
  double input_memory_usage = 0.;
  for (const auto& input : inputs_) {
    input_memory_usage += input->GetMemoryUsage();
  }
  return self_memory_usage + input_memory_usage;
}

double AutotuneNode::GetInactiveTimeMs() const {
  absl::MutexLock lock(&timing_stats_mutex_);
  return timing_stats_.mean_inactive_time_estimator.GetEstimate();
}

double AutotuneNode::GetConsumerTimeMs() const {
  if (output_.lock() == nullptr) {
    return GetInactiveTimeMs();
  }
  double output_consumer_time_ms = output_.lock()->GetConsumerTimeMs();
  double output_self_time_ms = output_.lock()->GetSelfTimeMs();
  double output_input_ratio = output_.lock()->GetInputRatio();
  return (output_consumer_time_ms + output_self_time_ms) / output_input_ratio;
}

absl::flat_hash_map<int64_t, double> AutotuneNode::ComputeParameterGradients()
    const {
  absl::flat_hash_map<int64_t, double> gradients;
  // Initialize previous input partial derivative to 1.
  ComputeParameterGradientsImpl(1., gradients);
  return gradients;
}

void AutotuneNode::ComputeParameterGradientsImpl(
    double prev_input_partial_derivative,
    absl::flat_hash_map<int64_t, double>& gradients) const {
  PartialDerivatives partial_derivatives = ComputePartialDerivatives();
  for (const auto& [param_id, derivative] : partial_derivatives.wrt_params) {
    gradients[param_id] += prev_input_partial_derivative * derivative;
  }
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->ComputeParameterGradientsImpl(
        prev_input_partial_derivative * partial_derivatives.wrt_inputs[i],
        gradients);
  }
}

AutotuneNode::PartialDerivatives
AutotuneNode::ComputeMemoryUsagePartialDerivatives() const {
  PartialDerivatives grads{
      .wrt_inputs = std::vector<double>(inputs_.size(), 1.0),
      .wrt_params = {},
  };
  return grads;
}

absl::flat_hash_map<int64_t, double> AutotuneNode::ComputeMemoryUsageGradients()
    const {
  absl::flat_hash_map<int64_t, double> gradients;
  // Initialize previous input partial derivative to 1.
  ComputeMemoryUsageGradientsImpl(1., gradients);
  return gradients;
}

void AutotuneNode::ComputeMemoryUsageGradientsImpl(
    double prev_input_partial_derivative,
    absl::flat_hash_map<int64_t, double>& gradients) const {
  PartialDerivatives partial_derivatives =
      ComputeMemoryUsagePartialDerivatives();
  for (const auto& [param_id, derivative] : partial_derivatives.wrt_params) {
    gradients[param_id] += prev_input_partial_derivative * derivative;
  }
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->ComputeMemoryUsageGradientsImpl(
        prev_input_partial_derivative * partial_derivatives.wrt_inputs[i],
        gradients);
  }
}

double AutotuneNode::GetElementSizeBytes() const {
  absl::MutexLock lock(&throughput_stats_mutex_);
  return throughput_stats_.mean_element_size_bytes_estimator.GetEstimate();
}

int64_t AutotuneNode::GetMinElementSizeBytes() const {
  absl::MutexLock lock(&throughput_stats_mutex_);
  return throughput_stats_.min_element_size_bytes;
}

int64_t AutotuneNode::GetMaxElementSizeBytes() const {
  absl::MutexLock lock(&throughput_stats_mutex_);
  return throughput_stats_.max_element_size_bytes;
}

std::vector<std::shared_ptr<AutotuneParameter>>
AutotuneNode::GetTunableParameters() const {
  std::vector<std::shared_ptr<AutotuneParameter>> params;
  GetLocalTunableParameters(params);
  for (const auto& input : inputs_) {
    std::vector<std::shared_ptr<AutotuneParameter>> input_params =
        input->GetTunableParameters();
    params.insert(params.end(), input_params.begin(), input_params.end());
  }
  return params;
}

void AutotuneNode::GetLocalTunableParameters(
    std::vector<std::shared_ptr<AutotuneParameter>>& params) const {}

void AutotuneNode::ConcurrencyModel::FlushPendingUpdates() {
  if (pending_rls_updates.empty()) {
    return;
  }
  for (const auto& pending : pending_rls_updates) {
    (void)usl_estimator.Update(pending.norm_x, pending.obs);
  }
  pending_rls_updates.clear();
  last_spectral_calc_obs_ = -1;
}

AutotuneNode::ConcurrencyModel& AutotuneNode::ConcurrencyModel::operator=(
    const ConcurrencyModel& other) {
  if (this != &other) {
    const_cast<ConcurrencyModel&>(other).FlushPendingUpdates();
    usl_estimator = other.usl_estimator;
    active_threads = other.active_threads;
    integral_active_threads_ms = other.integral_active_threads_ms;
    last_integral_time = other.last_integral_time;
    num_observations_ = other.num_observations_;
    history = other.history;
    concurrency_bins_seen = other.concurrency_bins_seen;
    spectral_activation_logged = other.spectral_activation_logged;
    pending_rls_updates.clear();
    cached_spectral_weight_ = other.cached_spectral_weight_;
    last_spectral_calc_obs_ = other.last_spectral_calc_obs_;
  }
  return *this;
}

double AutotuneNode::ConcurrencyModel::BaseThroughput() const {
  const_cast<ConcurrencyModel*>(this)->FlushPendingUpdates();
  double c2_norm = usl_estimator.GetEstimates()(2);
  double c2 = c2_norm / kScaleX2;
  if (c2 <= kMinUslPenalty) {
    return 0.0;
  }
  return 1.0 / c2;
}

double AutotuneNode::ConcurrencyModel::ComputeSpectralActivationWeight() const {
  if (num_observations_ < kMinObservationsForSpectralActivation) {
    return 0.0;
  }
  const_cast<ConcurrencyModel*>(this)->FlushPendingUpdates();
  if (num_observations_ == last_spectral_calc_obs_) {
    return cached_spectral_weight_;
  }
  // Information matrix R = P^-1
  Eigen::Matrix3d r = usl_estimator.GetInformationMatrix();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(r,
                                                        Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success) {
    return 0.0;
  }
  double lambda_min = solver.eigenvalues().minCoeff();
  // The Fisher Information Matrix is mathematically positive semi-definite,
  // but small numerical errors during eigensolving may yield a small negative
  // number.
  if (lambda_min <= 0.0) {
    return 0.0;
  }
  double exponent = -kKappa * (lambda_min - kTauInfo);
  double raw_w = 1.0 / (1.0 + std::exp(exponent));
  double w_0 = 1.0 / (1.0 + std::exp(kKappa * kTauInfo));
  double weight = std::max(0.0, (raw_w - w_0) / (1.0 - w_0));
  cached_spectral_weight_ = weight;
  last_spectral_calc_obs_ = num_observations_;
  return weight;
}

double AutotuneNode::ConcurrencyModel::Contention() const {
  const_cast<ConcurrencyModel*>(this)->FlushPendingUpdates();
  double c2_norm = usl_estimator.GetEstimates()(2);
  double c2 = c2_norm / kScaleX2;
  if (c2 <= kMinUslPenalty) {
    return 0.0;
  }
  double w = ComputeSpectralActivationWeight();
  double c1 = usl_estimator.GetEstimates()(1) / kScaleX1;
  return w * (c1 / c2);
}

double AutotuneNode::ConcurrencyModel::Coherency() const {
  const_cast<ConcurrencyModel*>(this)->FlushPendingUpdates();
  double c2_norm = usl_estimator.GetEstimates()(2);
  double c2 = c2_norm / kScaleX2;
  if (c2 <= kMinUslPenalty) {
    return 0.0;
  }
  double w = ComputeSpectralActivationWeight();
  double c0 = usl_estimator.GetEstimates()(0) / kScaleX0;
  return w * (c0 / c2);
}

double AutotuneNode::ConcurrencyModel::Evaluate(double n) const {
  double base_throughput = BaseThroughput();
  if (base_throughput <= 0.0) return 0.0;
  double alpha = Contention();
  double beta = Coherency();

  // usl_penalty is the Universal Scaling Law (USL) multiplier that models
  // contention (alpha) and coherency (beta) overhead. It acts as a performance
  // penalty factor applied to ideal parallel scaling. We apply the Softplus
  // function to smoothly clamp it above 0, preventing numerical instabilities
  // or negative latencies when the RLS estimator tests extreme coefficients.
  double overhead = alpha * (n - 1.0) + beta * n * (n - 1.0);
  double usl_penalty = 1.0 + Softplus(overhead);
  return base_throughput * n / usl_penalty;
}

void AutotuneNode::ClearInputs() {
  for (auto& input : inputs_) {
    input->output_.reset();
  }
  inputs_.clear();
}

std::vector<std::pair<double, double>> AutotuneNode::GetConcurrencyHistory()
    const {
  absl::MutexLock lock(&concurrency_model_mutex_);
  std::vector<std::pair<double, double>> result;
  result.reserve(concurrency_model_.history.size());
  for (const auto& obs : concurrency_model_.history) {
    result.push_back({obs.n, obs.obs_ms});
  }
  return result;
}

void AutotuneNode::ResetUslEstimator() {
  absl::MutexLock lock(&concurrency_model_mutex_);
  Eigen::Matrix<double, 3, 3> initial_cov =
      Eigen::Matrix<double, 3, 3>::Identity() * 1e4;
  initial_cov(0, 0) = 1.0;
  concurrency_model_.usl_estimator = RecursiveLeastSquares<3>::WithState(
      Eigen::Matrix<double, 3, 1>::Zero(), initial_cov, 0.999);
  concurrency_model_.history.clear();
  concurrency_model_.pending_rls_updates.clear();
  concurrency_model_.num_observations_ = 0;
  concurrency_model_.concurrency_bins_seen.reset();
  concurrency_model_.spectral_activation_logged = false;
  concurrency_model_.cached_spectral_weight_ = 0.0;
  concurrency_model_.last_spectral_calc_obs_ = -1;
}

int AutotuneNode::GetConcurrencyBinsSeen() const {
  absl::MutexLock lock(&concurrency_model_mutex_);
  return concurrency_model_.concurrency_bins_seen.count();
}

void AutotuneNode::TEST_RecordConcurrencyObservation(double n,
                                                     double elapsed_ms) {
  absl::MutexLock lock(&concurrency_model_mutex_);
  concurrency_model_.FlushPendingUpdates();
  double obs = elapsed_ms / n;
  concurrency_model_.history.push_back({n, obs});
  if (concurrency_model_.history.size() > 1000) {
    concurrency_model_.history.pop_front();
  }
  concurrency_model_.num_observations_++;
  int bin_idx;
  if (n < 4.0) {
    bin_idx = std::max<int>(0, static_cast<int>(std::floor(n)));
  } else {
    bin_idx =
        4 + std::max<int>(0, static_cast<int>(std::floor(std::log2(n / 4.0))));
  }
  bin_idx = std::min<int>(63, bin_idx);
  concurrency_model_.concurrency_bins_seen.set(bin_idx);
  double raw_x0 = n - 1.0;
  double raw_x1 = (n - 1.0) / n;
  double raw_x2 = 1.0 / n;
  Eigen::Vector3d norm_x(raw_x0 / ConcurrencyModel::kScaleX0,
                         raw_x1 / ConcurrencyModel::kScaleX1,
                         raw_x2 / ConcurrencyModel::kScaleX2);
  (void)concurrency_model_.usl_estimator.Update(norm_x, obs);
  concurrency_model_.last_spectral_calc_obs_ = -1;
}

}  // namespace grain::autotune
