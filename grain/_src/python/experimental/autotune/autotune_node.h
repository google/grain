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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_NODE_H_

#include <atomic>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"
#include "grain/_src/python/experimental/autotune/recursive_least_squares.h"

namespace grain::autotune {

// Base class for nodes in the autotune graph. This class is thread-safe.
//
// Derived types should define the input ratio (may be computed or static) as
// well as the timing model,
class AutotuneNode : public std::enable_shared_from_this<AutotuneNode> {
 public:
  // Thread-safe timing with per-thread state.
  class Timer {
   public:
    Timer();

    absl::StatusOr<absl::Duration> Start();
    absl::StatusOr<absl::Duration> End();
    absl::Status Pause();
    absl::Status Resume();
    int64_t NumPauses();
    void SetStartIntegral(double val) {
      GetState().start_integral_active_threads_ms = val;
    }
    double GetStartIntegral() {
      return GetState().start_integral_active_threads_ms;
    }

   private:
    struct TimerState {
      enum State {
        kInactive,
        kActive,
        kPaused,
      } state = kInactive;
      absl::Time time_start = absl::InfinitePast();
      absl::Duration time_elapsed = absl::ZeroDuration();
      absl::Time time_end = absl::InfinitePast();
      int64_t num_pauses = 0;
      double start_integral_active_threads_ms = 0.0;
    };
    TimerState& GetState();

    static inline thread_local std::vector<TimerState> registry_;
    static inline std::atomic<size_t> next_id_{0};
    const size_t id_;
  };

  // Timing stats aggregated across all threads.
  struct TimingStats {
    // The average time spent in the active state (excluding pauses).
    RecursiveLeastSquares<1> mean_latency_estimator =
        RecursiveLeastSquares<1>();
    // The average time spent in the inactive state.
    RecursiveLeastSquares<1> mean_inactive_time_estimator =
        RecursiveLeastSquares<1>();

    int64_t count = 0;
  };

  struct ThroughputStats {
    // The average size of the elements produced by this node.
    RecursiveLeastSquares<1> mean_element_size_bytes_estimator =
        RecursiveLeastSquares<1>();
    int64_t min_element_size_bytes = -1;
    int64_t max_element_size_bytes = -1;
  };

  struct ConcurrencyModel {
    ConcurrencyModel& operator=(const ConcurrencyModel& other);

    // This models throughput using the Universal Scaling Law (USL) as,
    //
    // X(N) = gamma * N / (1 + alpha * (N - 1) + beta * N * (N - 1))
    //
    // where:
    //   N is the number of active threads
    //   gamma is the base (serial) throughput
    //   alpha is the contention parameter
    //   beta is the coherency parameter
    //
    // where X(N) is the throughput with N threads. Note that 1 / X(N) is an
    // approximation of the nodes output time with N threads. For a
    // comprehensive explanation, see
    // https://www.perfdynamics.com/Manifesto/USLscalability.pdf.
    // BaseThroughput(), Contention(), and Coherency() are model parameters.
    // The Universal Scaling Law (USL) penalty term is defined as:
    //   usl_penalty = 1.0 + alpha * (N - 1) + beta * N * (N - 1)
    // To prevent numerical instability when the estimator drifts into extreme
    // or negative regions, we apply a Softplus function: Softplus(usl_penalty).
    // This acts as a lower-bounded smooth clamping above zero.
    double BaseThroughput() const;
    double Contention() const;
    double Coherency() const;
    double Evaluate(double n) const;
    double ComputeSpectralActivationWeight() const;

    RecursiveLeastSquares<3> usl_estimator = RecursiveLeastSquares<3>(0.999);
    // Has thread-local timer state.
    Timer timer;
    // TODO: b/487410476 - Use a weighted average to account for varying number
    // of active threads over the period of the nodes execution.
    int64_t active_threads = 0;
    double integral_active_threads_ms = 0.0;
    absl::Time last_integral_time;
    int64_t num_observations_ = 0;

    // History of observations for analysis and sweep logic.
    struct Observation {
      double n;
      double obs_ms;
    };
    std::deque<Observation> history;
    std::bitset<64> concurrency_bins_seen;
    bool spectral_activation_logged = false;

    struct PendingObservation {
      Eigen::Vector3d norm_x;
      double obs;
    };
    std::vector<PendingObservation> pending_rls_updates;
    static constexpr size_t kRlsBatchSize = 16;

    mutable double cached_spectral_weight_ = 0.0;
    mutable int64_t last_spectral_calc_obs_ = -1;

    void FlushPendingUpdates();

    static constexpr int64_t kMinObservationsForSpectralActivation = 30;
    static constexpr double kTauInfo = 0.5;
    static constexpr double kKappa = 5.0;

    // Coordinate scale factors for SIFt-RLS regressor normalization:
    //   x0 = (N - 1)      ~ O(4)
    //   x1 = (N - 1) / N  ~ O(1)
    //   x2 = 1 / N        ~ O(0.5)
    static constexpr double kScaleX0 = 4.0;
    static constexpr double kScaleX1 = 1.0;
    static constexpr double kScaleX2 = 0.5;
  };

  // Node-local/partial derivatives for the timing model.
  struct PartialDerivatives {
    // Derivatives of the output latency with respect to input latency.
    std::vector<double> wrt_inputs;
    // Derivatives of the output latency with respect to parameters if
    // applicable.
    absl::flat_hash_map<int64_t, double> wrt_params;
  };

  AutotuneNode(std::string name = "Unknown Node",
               double forgetting_factor = 0.99);

  virtual ~AutotuneNode() = default;

  absl::string_view GetName() const { return name_; }

  // Returns the ratio of the input size to the output size.
  // For example, a batch operation with batch size 8 will have a fixed input
  // ratio of 8.
  virtual double GetInputRatio() const = 0;

  // Creates a deep copy of this node.
  virtual std::shared_ptr<AutotuneNode> Copy() const = 0;

  // Ensures no cycles exist and only a single root is present.
  virtual absl::Status Validate() const;

  // Used to build pipelines from nodes. This node will now have shared
  // ownership of the input node. Returns a bad status if the input node already
  // has an output. Returns a self pointer to allow chaining, particularly in
  // Python.
  virtual absl::StatusOr<std::shared_ptr<AutotuneNode>> AddInput(
      std::shared_ptr<AutotuneNode> input);

  const std::vector<std::shared_ptr<AutotuneNode>>& GetInputs() const {
    return inputs_;
  }

  std::shared_ptr<AutotuneNode> GetOutput() const { return output_.lock(); }

  // Traverses the children of the tree and returns a deep copy of the sub-tree
  // with this node as the root.
  std::shared_ptr<AutotuneNode> GetSnapshot() const;

  // Copies statistics to the destination node.
  virtual void CopyStatsTo(AutotuneNode* dest) const;

  // Serializes the node and its children into a proto snapshot.
  virtual AutotuneNodeSnapshot GetSnapshotProto() const;

  // Methods for updating stats. The timer is state machine that will return a
  // bad status if the node is in the wrong state.
  virtual absl::StatusOr<absl::Duration> RecordStart();
  virtual absl::StatusOr<absl::Duration> RecordEnd();
  virtual absl::Status RecordPause();
  absl::Status RecordResume();
  absl::Status RecordElementSizeBytes(int64_t bytes);

  // Number of times this node has been timed.
  int64_t Count() const {
    absl::MutexLock lock(&timing_stats_mutex_);
    return timing_stats_.count;
  }

  // Helper methods to get the most relevant stats for modelling.
  // The estimate of this nodes self processing time in ms excluding its inputs.
  double GetSelfTimeMs() const;
  // The estimate of this nodes self memory usage in bytes excluding its inputs.
  virtual double GetSelfMemoryUsage() const;
  // The total aggregate time of the pipeline up to and including this node in
  // ms. Must be overridden by children to correctly account for how the inputs
  // are consumed.
  virtual double GetOutputTimeMs() const;

  // The total aggregate memory usage of the pipeline up to and including this
  // node in bytes. Must be overridden by children to account for memory usage
  // of the node and its inputs.
  virtual double GetMemoryUsage() const;

  // The average time the timer spends in the inactive state.
  double GetInactiveTimeMs() const;

  // The average time between successive requests to this node ignoring the time
  // spent in this node or its inputs. For the output node, this is just the
  // time between successive requests to the pipeline or the inactive time.
  double GetConsumerTimeMs() const;

  // The average size of the elements produced by this node.
  double GetElementSizeBytes() const;
  int64_t GetMinElementSizeBytes() const;
  int64_t GetMaxElementSizeBytes() const;

  // Concurrency model stats.
  double GetBaseThroughput() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.BaseThroughput();
  }
  double GetContention() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.Contention();
  }
  double GetCoherency() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.Coherency();
  }
  double GetBaseThroughputVariance() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.usl_estimator.GetCovariance()(2, 2);
  }
  double GetContentionVariance() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.usl_estimator.GetCovariance()(1, 1);
  }
  double GetCoherencyVariance() const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.usl_estimator.GetCovariance()(0, 0);
  }
  double GetModelThroughput(double n) const {
    absl::MutexLock lock(&concurrency_model_mutex_);
    return concurrency_model_.Evaluate(n);
  }

  std::vector<std::pair<double, double>> GetConcurrencyHistory() const;
  int GetConcurrencyBinsSeen() const;

  void ResetUslEstimator();

  // This is mostly for testing purposes where the graph may be intentionally in
  // a bad state, e.g. a cycle. Without this call there is a memory leak.
  void ClearInputs();

  // Recursively return all parameters from this node and its upstream inputs.
  virtual std::vector<std::shared_ptr<AutotuneParameter>> GetTunableParameters()
      const;

  // Return only the parameters owned directly by this node.
  virtual void GetLocalTunableParameters(
      std::vector<std::shared_ptr<AutotuneParameter>>& params) const;

  // Compute the local gradients for this node.
  virtual PartialDerivatives ComputePartialDerivatives() const = 0;

  // Computes the gradients of this nodes output time w.r.t. tunable parameters
  // using back-propagation. The map keys are the parameters unique integer ids.
  absl::flat_hash_map<int64_t, double> ComputeParameterGradients() const;

  // Computes the local gradients for global constraints.
  virtual PartialDerivatives ComputeMemoryUsagePartialDerivatives() const;

  // Computes the gradients of this nodes memory usage w.r.t. tunable
  // parameters using back-propagation. The map keys are the parameters unique
  // integer ids.
  absl::flat_hash_map<int64_t, double> ComputeMemoryUsageGradients() const;

  // This is used for testing purposes.
  void SetTimerStats(const TimingStats& timing_stats) {
    absl::MutexLock lock(&timing_stats_mutex_);
    timing_stats_ = timing_stats;
  }

  void SetThroughputStats(const ThroughputStats& throughput_stats) {
    absl::MutexLock lock(&throughput_stats_mutex_);
    throughput_stats_ = throughput_stats;
  }

  void SetConcurrencyStats(const RecursiveLeastSquares<3>& usl_estimator,
                           int64_t num_observations,
                           const std::bitset<64>& concurrency_bins_seen = {}) {
    absl::MutexLock lock(&concurrency_model_mutex_);
    concurrency_model_.usl_estimator = usl_estimator;
    concurrency_model_.num_observations_ = num_observations;
    concurrency_model_.concurrency_bins_seen = concurrency_bins_seen;
    concurrency_model_.pending_rls_updates.clear();
    concurrency_model_.cached_spectral_weight_ = 0.0;
    concurrency_model_.last_spectral_calc_obs_ = -1;
  }

  // Whether or not this node is an async node.
  virtual bool IsAsync() const { return false; }

  void SetConcurrencyModelForgettingFactor(double factor) {
    absl::MutexLock lock(&concurrency_model_mutex_);
    concurrency_model_.usl_estimator = RecursiveLeastSquares<3>(factor);
    concurrency_model_.pending_rls_updates.clear();
    concurrency_model_.cached_spectral_weight_ = 0.0;
    concurrency_model_.last_spectral_calc_obs_ = -1;
  }

  // Methods for testing concurrency model.
  void TEST_SetNumActiveThreads(int n) {
    absl::MutexLock lock(&concurrency_model_mutex_);
    concurrency_model_.active_threads = n;
  }
  void TEST_RecordConcurrencyObservation(double n, double elapsed_ms);

 protected:
  // Traverses the pipeline model and computes the parameter gradients.
  void ComputeParameterGradientsImpl(
      double prev_input_partial_derivative,
      absl::flat_hash_map<int64_t, double>& gradients) const;

  // Traverses the pipeline model and computes the memory usage gradients.
  void ComputeMemoryUsageGradientsImpl(
      double prev_input_partial_derivative,
      absl::flat_hash_map<int64_t, double>& gradients) const;

  std::string name_;

  // Producers of simulated elements for this node. The leaves will have no
  // inputs.
  std::vector<std::shared_ptr<AutotuneNode>> inputs_;
  // The downstream consumer of the nodes simulated output. The root node will
  // have no output.
  std::weak_ptr<AutotuneNode> output_;

  Timer timer_;
  mutable absl::Mutex timing_stats_mutex_;
  TimingStats timing_stats_ ABSL_GUARDED_BY(timing_stats_mutex_);
  mutable absl::Mutex throughput_stats_mutex_;
  ThroughputStats throughput_stats_ ABSL_GUARDED_BY(throughput_stats_mutex_);
  mutable absl::Mutex concurrency_model_mutex_;
  ConcurrencyModel concurrency_model_ ABSL_GUARDED_BY(concurrency_model_mutex_);
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_NODE_H_
