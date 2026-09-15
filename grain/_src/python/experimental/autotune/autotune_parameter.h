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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_PARAMETER_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_PARAMETER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace grain::autotune {

// Thread-safe class for holding a parameter used by an autotune node for
// modelling, and by grain dataset iterators for adjusting parallelism/buffer
// sizes etc. This class is thread-safe.
class AutotuneParameter {
 public:
  AutotuneParameter(std::string name,
                    std::optional<double> initial_value = std::nullopt,
                    std::optional<double> min_value = std::nullopt,
                    std::optional<double> max_value = std::nullopt,
                    double trust_region_multiplier = 2.0);

  absl::string_view GetName() const { return name_; }

  int64_t id() const { return id_; }

  double GetValue() const;

  double GetTrustRegionMultiplier() const { return trust_region_multiplier_; }

  const std::pair<double, double>& GetRange() const { return valid_range_; }

  absl::Status SetValue(double value, bool validate_value = true);

  operator double() const { return GetValue(); }

 private:
  std::string name_;
  // Unique integer id for this parameter.
  int64_t id_;
  std::pair<double, double> valid_range_;

  double value_;
  double trust_region_multiplier_;
};

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_AUTOTUNE_PARAMETER_H_
