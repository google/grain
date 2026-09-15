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

#include "grain/_src/python/experimental/autotune/autotune_model_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(__unix__)
#include <unistd.h>
#endif

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/time/time.h"

namespace grain::autotune {
namespace {
// The relationship between logical and physical memory can be impacted
// by fragmentation in tcmalloc. Leave some headroom for that.
constexpr double kAllowedMemoryUtilization = 0.8;
constexpr int64_t kGiB = 1024LL * 1024 * 1024;
}  // namespace

AutotuneModelConfig::AutotuneModelConfig(std::optional<size_t> cpu_budget,
                                         std::optional<double> ram_budget_gb,
                                         int optimization_frequency,
                                         int warmup_steps)
    : cpu_budget_(cpu_budget),
      ram_budget_gb_(ram_budget_gb),
      optimization_frequency_(optimization_frequency),
      warmup_steps_(warmup_steps)
{}

absl::StatusOr<size_t> AutotuneModelConfig::EstimateCpuAvailability() const {
  if (auto cgroup2_est = EstimateCpuAvailabilityCgroupV2();
      cgroup2_est.ok() && *cgroup2_est > 0) {
    return cgroup2_est;
  }
  if (auto cgroup1_est = EstimateCpuAvailabilityCgroupV1();
      cgroup1_est.ok() && *cgroup1_est > 0) {
    return cgroup1_est;
  }

  unsigned int hw_threads = std::thread::hardware_concurrency();
  return hw_threads > 0 ? static_cast<size_t>(hw_threads) : 4;
}

std::string AutotuneModelConfig::ReadFileToString(
    const std::string& path) const {
  std::ifstream file(path);
  if (!file.is_open()) {
    return "";
  }
  std::string content;
  std::string line;
  while (std::getline(file, line)) {
    absl::StrAppend(&content, line, "\n");
  }
  return content;
}

absl::StatusOr<double> AutotuneModelConfig::EstimateRamAvailabilityCgroupV2()
    const {
  std::optional<size_t> cgroup_limit;

  // Cgroup v2
  // https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory-interface-files
  std::string limit_str = ReadFileToString("/sys/fs/cgroup/memory.max");
  limit_str = std::string(absl::StripTrailingAsciiWhitespace(limit_str));
  if (!limit_str.empty() && limit_str != "max") {
    size_t limit_bytes;
    if (absl::SimpleAtoi(limit_str, &limit_bytes)) {
      std::string usage_str = ReadFileToString("/sys/fs/cgroup/memory.current");
      usage_str = std::string(absl::StripTrailingAsciiWhitespace(usage_str));
      size_t usage_bytes;
      if (absl::SimpleAtoi(usage_str, &usage_bytes)) {
        if (limit_bytes > usage_bytes) {
          cgroup_limit = limit_bytes - usage_bytes;
        }
      }
    }
  }
  if (cgroup_limit.has_value()) {
    double available_gb = static_cast<double>(*cgroup_limit) / kGiB;
    return available_gb * kAllowedMemoryUtilization;
  }
  return absl::NotFoundError("CgroupV2 RAM: memory.max not found or 'max'.");
}

absl::StatusOr<double> AutotuneModelConfig::EstimateRamAvailabilityCgroupV1()
    const {
  std::optional<size_t> cgroup_limit;
  // Cgroup v1
  // https://www.kernel.org/doc/Documentation/cgroup-v1/memory.txt
  std::string limit_str =
      ReadFileToString("/sys/fs/cgroup/memory/memory.limit_in_bytes");
  limit_str = std::string(absl::StripTrailingAsciiWhitespace(limit_str));
  size_t limit_bytes;
  if (absl::SimpleAtoi(limit_str, &limit_bytes)) {
    if (limit_bytes < (1LL << 60)) {  // Cgroups v1 does not use "max".
      std::string usage_str =
          ReadFileToString("/sys/fs/cgroup/memory/memory.usage_in_bytes");
      usage_str = std::string(absl::StripTrailingAsciiWhitespace(usage_str));
      size_t usage_bytes;
      if (absl::SimpleAtoi(usage_str, &usage_bytes)) {
        if (limit_bytes > usage_bytes) {
          cgroup_limit = limit_bytes - usage_bytes;
        }
      }
    }
  }

  if (cgroup_limit.has_value()) {
    double available_gb = static_cast<double>(*cgroup_limit) / kGiB;
    return available_gb * kAllowedMemoryUtilization;
  }
  return absl::NotFoundError("CgroupV1 RAM: memory.limit_in_bytes not found.");
}

absl::StatusOr<double> AutotuneModelConfig::EstimateRamAvailabilityGb() const {
  if (auto cgroup_est = EstimateRamAvailabilityCgroupV2();
      cgroup_est.ok() && *cgroup_est > 0) {
    return cgroup_est;
  }
  if (auto cgroup_est = EstimateRamAvailabilityCgroupV1();
      cgroup_est.ok() && *cgroup_est > 0) {
    return cgroup_est;
  }
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  if (GlobalMemoryStatusEx(&status)) {
    double total_gb = static_cast<double>(status.ullTotalPhys) / kGiB;
    return total_gb * kAllowedMemoryUtilization;
  }
#elif defined(__APPLE__)
  int64_t memsize = 0;
  size_t len = sizeof(memsize);
  if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0 &&
      memsize > 0) {
    double total_gb = static_cast<double>(memsize) / kGiB;
    return total_gb * kAllowedMemoryUtilization;
  }
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    double total_gb = (static_cast<double>(pages) * page_size) / kGiB;
    return total_gb * kAllowedMemoryUtilization;
  }
#endif
  return 4.0 * kAllowedMemoryUtilization;
}

absl::StatusOr<double> AutotuneModelConfig::GetRamBudgetGb() {
  if (!ram_budget_gb_.has_value()) {
    auto est = EstimateRamAvailabilityGb();
    if (est.ok()) {
      ram_budget_gb_ = *est;
    } else {
      return est.status();
    }
  }
  return ram_budget_gb_.value();
}

absl::StatusOr<size_t> AutotuneModelConfig::GetCpuBudget() {
  if (!cpu_budget_.has_value()) {
    auto est = EstimateCpuAvailability();
    if (est.ok()) {
      cpu_budget_ = *est;
    } else {
      return est.status();
    }
  }
  return cpu_budget_.value();
}

absl::StatusOr<size_t> AutotuneModelConfig::EstimateCpuAvailabilityCgroupV2()
    const {
  // Cgroup v2
  // https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#cpu-interface-files
  std::string limit_content = ReadFileToString("/sys/fs/cgroup/cpu.max");
  if (limit_content.empty()) {
    return absl::NotFoundError(
        "CgroupV2 CPU: /sys/fs/cgroup/cpu.max could not be opened or is "
        "empty.");
  }
  std::vector<std::string> parts =
      absl::StrSplit(absl::StripTrailingAsciiWhitespace(limit_content), ' ',
                     absl::SkipEmpty());
  if (parts.empty()) {
    return absl::InternalError("CgroupV2 CPU: cpu.max content is invalid.");
  }
  std::string quota_str = parts[0];
  if (quota_str == "max") {
    return absl::UnavailableError(
        "CgroupV2 CPU: quota is 'max', meaning no limit.");
  }
  if (parts.size() >= 2) {
    std::string period_str = parts[1];
    size_t quota_us = 0;
    size_t period_us = 0;
    if (absl::SimpleAtoi(quota_str, &quota_us) &&
        absl::SimpleAtoi(period_str, &period_us)) {
      if (period_us > 0) {
        return std::max<size_t>(
            1, std::ceil(static_cast<double>(quota_us) / period_us));
      }
    }
  }
  return absl::InternalError("CgroupV2 CPU: failed to parse quota or period.");
}

absl::StatusOr<size_t> AutotuneModelConfig::EstimateCpuAvailabilityCgroupV1()
    const {
  // Cgroup v1
  // https://www.kernel.org/doc/Documentation/scheduler/sched-bwc.txt
  std::string quota_str =
      ReadFileToString("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
  if (quota_str.empty()) {
    return absl::NotFoundError(
        "CgroupV1 CPU: /sys/fs/cgroup/cpu/cpu.cfs_quota_us could not be opened "
        "or is empty.");
  }
  int64_t quota_us;
  if (absl::SimpleAtoi(absl::StripTrailingAsciiWhitespace(quota_str),
                       &quota_us)) {
    if (quota_us > 0) {
      std::string period_str =
          ReadFileToString("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
      if (period_str.empty()) {
        return absl::NotFoundError(
            "CgroupV1 CPU: /sys/fs/cgroup/cpu/cpu.cfs_period_us could not be "
            "opened or is empty.");
      }
      size_t period_us;
      if (absl::SimpleAtoi(absl::StripTrailingAsciiWhitespace(period_str),
                           &period_us) &&
          period_us > 0) {
        return std::max<size_t>(
            1, std::ceil(static_cast<double>(quota_us) / period_us));
      }
    } else {
      return absl::UnavailableError(absl::StrCat("CgroupV1 CPU: quota_us is ",
                                                 quota_us,
                                                 " (<= 0), meaning no limit."));
    }
  }
  return absl::InternalError("CgroupV1 CPU: failed to parse quota.");
}

}  // namespace grain::autotune
