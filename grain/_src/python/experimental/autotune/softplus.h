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

#ifndef THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOFTPLUS_H_
#define THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOFTPLUS_H_

namespace grain::autotune {

// Numerically stable softplus function: Softplus(x) = log(1 + exp(x)).
// This provides a smooth approximation to the ReLU function and is used to
// ensure numerical stability by providing a lower bound on the penalty term
// in the Universal Scalability Law (USL) model.
double Softplus(double x);

// Derivative of the softplus function, which is the logistic function:
// SoftplusDerivative(x) = 1 / (1 + exp(-x)).
double SoftplusDerivative(double x);

}  // namespace grain::autotune

#endif  // THIRD_PARTY_PY_GRAIN__SRC_PYTHON_EXPERIMENTAL_AUTOTUNE_SOFTPLUS_H_
