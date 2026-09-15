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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_ASYNC_FIXED_RATIO_NODE_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_ASYNC_FIXED_RATIO_NODE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"

namespace grain::autotune {

class AsyncFixedRatioNode : public AutotuneNode {
 public:
  AsyncFixedRatioNode(
      double input_ratio = 1.0, std::string name = "",
      double forgetting_factor = 0.99,
      std::variant<double, std::shared_ptr<AutotuneParameter>> concurrency =
          1.0,
      std::variant<double, std::shared_ptr<AutotuneParameter>> buffer_size =
          1.0);

  // Overrides.
  double GetInputRatio() const override { return input_ratio_; }
  std::shared_ptr<AutotuneNode> Copy() const override;
  AutotuneNodeSnapshot GetSnapshotProto() const override;

  // This returns the wait time computed by the queuing model.
  double GetOutputTimeMs() const override;
  // This returns the estimated self memory usage of the node in bytes.
  double GetSelfMemoryUsage() const override;
  // This returns the estimated memory usage of the node in bytes.
  double GetMemoryUsage() const override;
  PartialDerivatives ComputePartialDerivatives() const override;
  PartialDerivatives ComputeMemoryUsagePartialDerivatives() const override;

  // Helper methods.
  double GetConcurrency() const;
  double GetBufferSize() const;

  // Common metric in queuing theory. Measures the ratio of the arrival rate to
  // the service rate, in this case it is the ratio of producer and consumer
  // rates.
  double TrafficIntensity() const;

  // Derivative of wait time w.r.t. traffic intensity.
  double TrafficIntensitySensitivity() const;

  // Derivative of wait time w.r.t. buffer size.
  double BufferSizeSensitivity() const;

  // Derivative of traffic intensity w.r.t. concurrency. Depends on the model
  // used to model parallelism.
  double TrafficIntensityWrtConcurrencySensitivity() const;

  std::vector<double> ArrivalRateSensitivityWrtInputTime() const;

  double GetProducerTimeMsIdeal() const;
  double GetProducerTimeMs() const;
  double GetProducerTimeMsWrtConcurrencySensitivity() const;

  bool IsAsync() const override { return true; }

  enum class BufferRegularizationMode {
    kNone = 0,
    kL2 = 1,
    kBarrier = 2,
  };

  void SetBufferRegularization(BufferRegularizationMode mode, double weight) {
    buffer_reg_mode_ = mode;
    buffer_reg_weight_ = weight;
  }
  BufferRegularizationMode GetBufferRegularizationMode() const {
    return buffer_reg_mode_;
  }
  double GetBufferRegularizationWeight() const { return buffer_reg_weight_; }

  enum class ConcurrencyRegularizationMode {
    kNone = 0,
    kLinear = 1,
    kQuadratic = 2,
  };

  void SetConcurrencyRegularization(ConcurrencyRegularizationMode mode,
                                    double weight) {
    concurrency_reg_mode_ = mode;
    concurrency_reg_weight_ = weight;
  }
  ConcurrencyRegularizationMode GetConcurrencyRegularizationMode() const {
    return concurrency_reg_mode_;
  }
  double GetConcurrencyRegularizationWeight() const {
    return concurrency_reg_weight_;
  }

 private:
  static constexpr double kBufferContentionFactor = 0.20;

  static double GetConcurrencyMemoryUsage(double memory_usage_inputs,
                                          double concurrency);

  double GetMaxConcurrency() const;

  void GetLocalTunableParameters(
      std::vector<std::shared_ptr<AutotuneParameter>>& parameters)
      const override;

  const double input_ratio_;

  // Concurrency.
  std::variant<double, std::shared_ptr<AutotuneParameter>> n_;
  // Buffer size.
  std::variant<double, std::shared_ptr<AutotuneParameter>> k_;

  BufferRegularizationMode buffer_reg_mode_ = BufferRegularizationMode::kNone;
  double buffer_reg_weight_ = 0.0;

  ConcurrencyRegularizationMode concurrency_reg_mode_ =
      ConcurrencyRegularizationMode::kNone;
  double concurrency_reg_weight_ = 0.0;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_ASYNC_FIXED_RATIO_NODE_H_
