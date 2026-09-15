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

#include "grain/_src/python/experimental/autotune/softplus.h"

#include <algorithm>
#include <cmath>

namespace grain::autotune {

namespace {
// Scale factor for softplus to control the sharpness of the transition.
constexpr double kSoftplusScale = 1e-3;
}  // namespace

double Softplus(double x) {
  // Numerically stable identity: Softplus(x) = max(0, x) + log(1 + exp(-|x|))
  // With scale s: s * log(1 + exp(x/s)) = max(0, x) + s * log(1 + exp(-|x|/s))
  return std::max(0.0, x) +
         kSoftplusScale * std::log1p(std::exp(-std::abs(x) / kSoftplusScale));
}

double SoftplusDerivative(double x) {
  // Logistic function: 1 / (1 + exp(-x/s))
  // For x > 0: 1 / (1 + exp(-x/s))
  // For x <= 0: exp(x/s) / (1 + exp(x/s))
  if (x > 0.0) {
    return 1.0 / (1.0 + std::exp(-x / kSoftplusScale));
  }
  const double exp_x = std::exp(x / kSoftplusScale);
  return exp_x / (1.0 + exp_x);
}

}  // namespace grain::autotune
