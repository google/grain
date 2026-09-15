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

#include "grain/_src/python/experimental/autotune/autotune_serialization.h"

#include <bitset>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "grain/_src/python/experimental/autotune/async_fixed_ratio_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"
#include "grain/_src/python/experimental/autotune/fixed_ratio_node.h"
#include "grain/_src/python/experimental/autotune/interleave_node.h"
#include "grain/_src/python/experimental/autotune/recursive_least_squares.h"
#include "grain/_src/python/experimental/autotune/source_node.h"
#include "grain/_src/python/experimental/autotune/unknown_ratio_node.h"

namespace grain::autotune {

namespace {

template <int NumParameters>
RecursiveLeastSquares<NumParameters> RecoverRlsFilter(
    const AutotuneNodeSnapshot::RlsStats& rls_stats) {
  if (rls_stats.estimates_size() < NumParameters ||
      rls_stats.covariance_size() < NumParameters * NumParameters) {
    return RecursiveLeastSquares<NumParameters>(rls_stats.forgetting_factor());
  }

  typename RecursiveLeastSquares<NumParameters>::VectorType estimates;
  for (int i = 0; i < NumParameters; ++i) {
    estimates(i) = rls_stats.estimates(i);
  }

  typename RecursiveLeastSquares<NumParameters>::MatrixType covariance;
  for (int i = 0; i < NumParameters; ++i) {
    for (int j = 0; j < NumParameters; ++j) {
      covariance(i, j) = rls_stats.covariance(i * NumParameters + j);
    }
  }

  return RecursiveLeastSquares<NumParameters>::WithState(
      estimates, covariance, rls_stats.forgetting_factor());
}

}  // namespace

absl::StatusOr<std::shared_ptr<AutotuneNode>> RecoverAutotuneNode(
    const AutotuneNodeSnapshot& snapshot) {
  std::shared_ptr<AutotuneNode> node;

  switch (snapshot.node_type_case()) {
    case AutotuneNodeSnapshot::kSourceNode:
      node = std::make_shared<SourceNode>(snapshot.name(),
                                          snapshot.forgetting_factor());
      break;
    case AutotuneNodeSnapshot::kFixedRatioNode: {
      double ratio = snapshot.fixed_ratio_node().input_ratio();
      node = std::make_shared<FixedRatioNode>(ratio, snapshot.name(),
                                              snapshot.forgetting_factor());
      break;
    }
    case AutotuneNodeSnapshot::kUnknownRatioNode: {
      double ratio = snapshot.unknown_ratio_node().current_ratio();
      node = std::make_shared<UnknownRatioNode>(ratio, snapshot.name(),
                                                snapshot.forgetting_factor());
      break;
    }
    case AutotuneNodeSnapshot::kAsyncFixedRatioNode: {
      const auto& async_node = snapshot.async_fixed_ratio_node();
      node = std::make_shared<AsyncFixedRatioNode>(
          async_node.input_ratio(), snapshot.name(),
          snapshot.forgetting_factor(), async_node.concurrency(),
          async_node.buffer_size());
      break;
    }
    case AutotuneNodeSnapshot::kInterleaveNode: {
      int cycle_length = snapshot.interleave_node().cycle_length();
      if (cycle_length == 0) cycle_length = 1;
      node = std::make_shared<InterleaveNode>(
          snapshot.name(), snapshot.forgetting_factor(),
          static_cast<double>(cycle_length), 1.0);
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported node type case: %d", snapshot.node_type_case()));
  }

  // Restore stats
  if (snapshot.has_timing_stats()) {
    const auto& timing = snapshot.timing_stats();
    AutotuneNode::TimingStats stats;

    if (timing.has_mean_latency()) {
      stats.mean_latency_estimator = RecoverRlsFilter<1>(timing.mean_latency());
    }
    if (timing.has_mean_inactive_time()) {
      stats.mean_inactive_time_estimator =
          RecoverRlsFilter<1>(timing.mean_inactive_time());
    }
    stats.count = timing.count();
    node->SetTimerStats(stats);
  }

  if (snapshot.has_throughput_stats()) {
    const auto& throughput = snapshot.throughput_stats();
    AutotuneNode::ThroughputStats stats;
    if (throughput.has_mean_element_size_bytes()) {
      stats.mean_element_size_bytes_estimator =
          RecoverRlsFilter<1>(throughput.mean_element_size_bytes());
    }
    stats.min_element_size_bytes = throughput.min_element_size_bytes();
    stats.max_element_size_bytes = throughput.max_element_size_bytes();
    node->SetThroughputStats(stats);
  }

  if (snapshot.has_concurrency_stats()) {
    const auto& concurrency = snapshot.concurrency_stats();
    if (concurrency.has_usl_estimator()) {
      auto usl_estimator = RecoverRlsFilter<3>(concurrency.usl_estimator());
      std::bitset<64> concurrency_bins_seen(
          concurrency.concurrency_bins_seen());
      node->SetConcurrencyStats(usl_estimator, concurrency.num_observations(),
                                concurrency_bins_seen);
    }
  }

  // Recover inputs recursively
  for (const auto& input_snapshot : snapshot.inputs()) {
    auto input_or_status = RecoverAutotuneNode(input_snapshot);
    if (!input_or_status.ok()) return input_or_status.status();
    auto added_input_or_status = node->AddInput(*input_or_status);
    if (!added_input_or_status.ok()) {
      return added_input_or_status.status();
    }
  }

  return node;
}

}  // namespace grain::autotune
