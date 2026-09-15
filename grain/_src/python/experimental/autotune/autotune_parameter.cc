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

#include "grain/_src/python/experimental/autotune/autotune_parameter.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace grain::autotune {
namespace {
int64_t GetId() {
  static std::atomic<int64_t> next_id = 0;
  return next_id++;
}
}  // namespace

AutotuneParameter::AutotuneParameter(std::string name,
                                     std::optional<double> initial_value,
                                     std::optional<double> min_value,
                                     std::optional<double> max_value,
                                     double trust_region_multiplier)
    : name_(std::move(name)),
      id_(GetId()),
      valid_range_(min_value.value_or(1.0),
                   max_value.value_or(std::numeric_limits<double>::max())),
      value_(initial_value.value_or(valid_range_.first)),
      trust_region_multiplier_(trust_region_multiplier) {
  CHECK_LE(valid_range_.first, valid_range_.second);
}

double AutotuneParameter::GetValue() const {
  return value_;
}

absl::Status AutotuneParameter::SetValue(double value, bool validate_value) {
  if (validate_value) {
    if (value < valid_range_.first || value > valid_range_.second) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Value %f is not in range [%f, %f]", value,
                          valid_range_.first, valid_range_.second));
    }
  }
  value_ = value;
  return absl::OkStatus();
}

}  // namespace grain::autotune
