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

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/str_format.h"
#include "grain/_src/python/experimental/autotune/async_fixed_ratio_node.h"
#include "grain/_src/python/experimental/autotune/autotune_model.h"
#include "grain/_src/python/experimental/autotune/autotune_model_config.h"
#include "grain/_src/python/experimental/autotune/autotune_node.h"
#include "grain/_src/python/experimental/autotune/autotune_node.pb.h"
#include "grain/_src/python/experimental/autotune/autotune_parameter.h"
#include "grain/_src/python/experimental/autotune/autotune_serialization.h"
#include "grain/_src/python/experimental/autotune/fixed_ratio_node.h"
#include "grain/_src/python/experimental/autotune/interleave_node.h"
#include "grain/_src/python/experimental/autotune/source_node.h"
#include "grain/_src/python/experimental/autotune/unknown_ratio_node.h"
namespace pybind11::detail {
template <>
struct type_caster<absl::Status> {
 public:
  PYBIND11_TYPE_CASTER(absl::Status, _("Status"));
  static handle cast(absl::Status src, return_value_policy, handle) {
    pybind11::gil_scoped_acquire gil;
    if (!src.ok()) {
      throw std::runtime_error(std::string(src.message()));
    }
    return none().inc_ref();
  }
};

template <typename T>
struct type_caster<absl::StatusOr<T>> {
 public:
  using value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(absl::StatusOr<T>,
                       _("StatusOr[") + value_conv::name + _("]"));
  static handle cast(absl::StatusOr<T> src, return_value_policy policy,
                     handle parent) {
    pybind11::gil_scoped_acquire gil;
    if (!src.ok()) {
      throw std::runtime_error(std::string(src.status().message()));
    }
    return value_conv::cast(std::forward<T>(*src), policy, parent);
  }
};

template <>
struct type_caster<absl::Duration> {
 public:
  PYBIND11_TYPE_CASTER(absl::Duration, _("Duration"));
  static handle cast(absl::Duration src, return_value_policy, handle) {
    pybind11::gil_scoped_acquire gil;
    return float_(absl::ToDoubleSeconds(src)).inc_ref();
  }
};
}  // namespace pybind11::detail

namespace grain::autotune {
namespace {
namespace py = pybind11;

py::dict GetPipelineStats(const std::shared_ptr<AutotuneNode>& node) {
  py::dict stats;
  stats["name"] = std::string(node->GetName());
  stats["input_ratio"] = node->GetInputRatio();
  stats["self_time_ms"] = node->GetSelfTimeMs();
  stats["output_time_ms"] = node->GetOutputTimeMs();
  stats["consumer_time_ms"] = node->GetConsumerTimeMs();
  stats["count"] = node->Count();
  stats["element_size_bytes"] = node->GetElementSizeBytes();
  stats["min_element_size_bytes"] = node->GetMinElementSizeBytes();
  stats["max_element_size_bytes"] = node->GetMaxElementSizeBytes();
  stats["base_throughput"] = node->GetBaseThroughput();
  stats["base_throughput_variance"] = node->GetBaseThroughputVariance();
  stats["contention"] = node->GetContention();
  stats["contention_variance"] = node->GetContentionVariance();
  stats["coherency"] = node->GetCoherency();
  stats["concurrency_bins_seen"] = node->GetConcurrencyBinsSeen();

  py::dict params_dict;
  std::vector<std::shared_ptr<AutotuneParameter>> local_params;
  node->GetLocalTunableParameters(local_params);
  for (const auto& param : local_params) {
    params_dict[param->GetName().data()] = param->GetValue();
  }
  stats["tunable_parameters"] = params_dict;

  py::list inputs;
  for (const auto& input : node->GetInputs()) {
    inputs.append(GetPipelineStats(input));
  }
  stats["inputs"] = inputs;
  return stats;
}

}  // namespace

PYBIND11_MODULE(bindings, m) {
  m.doc() = "Python bindings for the Autotune C++ library.";

  py::class_<AutotuneNode, std::shared_ptr<AutotuneNode>>(m, "AutotuneNode")
      .def("__str__", &AutotuneNode::GetName)
      .def("get_input_ratio", &AutotuneNode::GetInputRatio)
      .def("add_input", &AutotuneNode::AddInput)
      .def("clear_inputs", &AutotuneNode::ClearInputs)
      .def("get_inputs", &AutotuneNode::GetInputs)
      .def("get_output", &AutotuneNode::GetOutput)
      .def("record_start", &AutotuneNode::RecordStart)
      .def("record_end", &AutotuneNode::RecordEnd)
      .def("record_pause", &AutotuneNode::RecordPause)
      .def("record_resume", &AutotuneNode::RecordResume)
      .def("get_self_time_ms", &AutotuneNode::GetSelfTimeMs)
      .def("get_output_time_ms", &AutotuneNode::GetOutputTimeMs)
      .def("get_consumer_time_ms", &AutotuneNode::GetConsumerTimeMs)
      .def("record_element_size_bytes", &AutotuneNode::RecordElementSizeBytes)
      .def("get_element_size_bytes", &AutotuneNode::GetElementSizeBytes)
      .def("get_min_element_size_bytes", &AutotuneNode::GetMinElementSizeBytes)
      .def("get_max_element_size_bytes", &AutotuneNode::GetMaxElementSizeBytes)
      .def("get_count", &AutotuneNode::Count)
      .def("get_pipeline_stats", &GetPipelineStats)
      .def("get_tunable_parameters",
           [](const AutotuneNode& node) { return node.GetTunableParameters(); })
      .def("get_concurrency_history", &AutotuneNode::GetConcurrencyHistory)
      .def("get_concurrency_bins_seen", &AutotuneNode::GetConcurrencyBinsSeen)
      .def("reset_usl_estimator", &AutotuneNode::ResetUslEstimator)
      .def("test_record_concurrency_observation",
           &AutotuneNode::TEST_RecordConcurrencyObservation)
      .def("get_local_tunable_parameters",
           [](const AutotuneNode& node) {
             std::vector<std::shared_ptr<AutotuneParameter>> params;
             node.GetLocalTunableParameters(params);
             return params;
           })
      .def("get_snapshot_bytes",
           [](const AutotuneNode& node) -> py::bytes {
             return node.GetSnapshotProto().SerializeAsString();
           })
      .def_property_readonly("is_async", &AutotuneNode::IsAsync);

  m.def("recover_autotune_node", [](py::bytes snapshot_bytes) {
    AutotuneNodeSnapshot snapshot;
    if (!snapshot.ParseFromString(snapshot_bytes)) {
      throw std::runtime_error("Failed to parse AutotuneNodeSnapshot");
    }
    return RecoverAutotuneNode(snapshot);
  });

  py::class_<SourceNode, AutotuneNode, std::shared_ptr<SourceNode>>(
      m, "SourceNode")
      .def(py::init<std::string, double>(), py::arg("name") = "SourceNode",
           py::arg("forgetting_factor") = 0.99);
  py::class_<FixedRatioNode, AutotuneNode, std::shared_ptr<FixedRatioNode>>(
      m, "FixedRatioNode")
      .def(py::init<double, std::string, double>(), py::arg("input_ratio"),
           py::arg("name") = "", py::arg("forgetting_factor") = 0.99);
  py::class_<InterleaveNode, AutotuneNode, std::shared_ptr<InterleaveNode>>(
      m, "InterleaveNode")
      .def(py::init<std::string, double,
                    std::variant<double, std::shared_ptr<AutotuneParameter>>,
                    double>(),
           py::arg("name") = "InterleaveNode",
           py::arg("forgetting_factor") = 0.99, py::arg("cycle_length") = 1.0,
           py::arg("make_iter_buffer_size") = 1.0);

  py::class_<UnknownRatioNode, AutotuneNode, std::shared_ptr<UnknownRatioNode>>(
      m, "UnknownRatioNode")
      .def(py::init<double, std::string, double>(),
           py::arg("initial_ratio") = 1.0, py::arg("name") = "UnknownRatioNode",
           py::arg("forgetting_factor") = 0.99)
      .def("record_ratio", &UnknownRatioNode::RecordRatio);

  py::enum_<AsyncFixedRatioNode::BufferRegularizationMode>(
      m, "BufferRegularizationMode")
      .value("NONE", AsyncFixedRatioNode::BufferRegularizationMode::kNone)
      .value("L2", AsyncFixedRatioNode::BufferRegularizationMode::kL2)
      .value("BARRIER", AsyncFixedRatioNode::BufferRegularizationMode::kBarrier)
      .export_values();

  py::enum_<AsyncFixedRatioNode::ConcurrencyRegularizationMode>(
      m, "ConcurrencyRegularizationMode")
      .value("NONE", AsyncFixedRatioNode::ConcurrencyRegularizationMode::kNone)
      .value("LINEAR",
             AsyncFixedRatioNode::ConcurrencyRegularizationMode::kLinear)
      .value("QUADRATIC",
             AsyncFixedRatioNode::ConcurrencyRegularizationMode::kQuadratic)
      .export_values();

  py::class_<AsyncFixedRatioNode, AutotuneNode,
             std::shared_ptr<AsyncFixedRatioNode>>(m, "AsyncFixedRatioNode")
      .def(py::init<double, std::string, double,
                    std::variant<double, std::shared_ptr<AutotuneParameter>>,
                    std::variant<double, std::shared_ptr<AutotuneParameter>>>(),
           py::arg("input_ratio") = 1.0,
           py::arg("name") = "AsyncFixedRatioNode",
           py::arg("forgetting_factor") = 0.99, py::arg("concurrency") = 1.0,
           py::arg("buffer_size") = 1.0)
      .def_property_readonly("concurrency",
                             &AsyncFixedRatioNode::GetConcurrency)
      .def_property_readonly("buffer_size", &AsyncFixedRatioNode::GetBufferSize)
      .def("set_buffer_regularization",
           &AsyncFixedRatioNode::SetBufferRegularization, py::arg("mode"),
           py::arg("weight"))
      .def("get_buffer_regularization_mode",
           &AsyncFixedRatioNode::GetBufferRegularizationMode)
      .def("get_buffer_regularization_weight",
           &AsyncFixedRatioNode::GetBufferRegularizationWeight)
      .def("set_concurrency_regularization",
           &AsyncFixedRatioNode::SetConcurrencyRegularization, py::arg("mode"),
           py::arg("weight"))
      .def("get_concurrency_regularization_mode",
           &AsyncFixedRatioNode::GetConcurrencyRegularizationMode)
      .def("get_concurrency_regularization_weight",
           &AsyncFixedRatioNode::GetConcurrencyRegularizationWeight);

  py::class_<AutotuneParameter, std::shared_ptr<AutotuneParameter>>(
      m, "AutotuneParameter")
      .def(py::init<std::string, std::optional<double>, std::optional<double>,
                    std::optional<double>, double>(),
           py::arg("name"), py::arg("initial_value") = std::nullopt,
           py::arg("min_value") = std::nullopt,
           py::arg("max_value") = std::nullopt,
           py::arg("trust_region_multiplier") = 2.0)
      .def("__str__",
           [](const AutotuneParameter& parameter) {
             return absl::StrFormat("%s (value = %f, range = [%f, %f])",
                                    parameter.GetName(), parameter.GetValue(),
                                    parameter.GetRange().first,
                                    parameter.GetRange().second);
           })
      .def("__int__",
           [](const AutotuneParameter& parameter) {
             return static_cast<int>(parameter.GetValue());
           })
      .def("__float__", &AutotuneParameter::GetValue)
      .def("get_value", &AutotuneParameter::GetValue)
      .def("set_value", &AutotuneParameter::SetValue, py::arg("value"),
           py::arg("validate_value") = true)
      .def_property_readonly("trust_region_multiplier",
                             &AutotuneParameter::GetTrustRegionMultiplier)
      .def_property_readonly("id", &AutotuneParameter::id)
      .def_property_readonly("name", &AutotuneParameter::GetName)
      .def(py::pickle(
          [](const AutotuneParameter& p) {  // __getstate__
            return py::make_tuple(p.GetName(), p.GetValue(), p.GetRange().first,
                                  p.GetRange().second,
                                  p.GetTrustRegionMultiplier());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 5) throw std::runtime_error("Invalid state!");

            return std::make_shared<AutotuneParameter>(
                t[0].cast<std::string>(), t[1].cast<double>(),
                t[2].cast<double>(), t[3].cast<double>(), t[4].cast<double>());
          }));

  py::class_<AutotuneModelConfig>(m, "AutotuneModelConfig")
      .def(py::init<std::optional<size_t>, std::optional<double>, int, int>(),
           py::arg("cpu_budget") = std::nullopt,
           py::arg("ram_budget_gb") = std::nullopt,
           py::arg("optimization_frequency") = 100, py::arg("warmup_steps") = 0)
      .def("get_cpu_budget", &AutotuneModelConfig::GetCpuBudget)
      .def("get_ram_budget_gb", &AutotuneModelConfig::GetRamBudgetGb)
      .def("get_optimization_frequency",
           &AutotuneModelConfig::GetOptimizationFrequency)
      .def("get_warmup_steps", &AutotuneModelConfig::GetWarmupSteps);

  py::class_<AutotuneModel>(m, "AutotuneModel")
      .def(py::init<AutotuneModelConfig>(), py::arg("model_config"))
      .def("optimize", &AutotuneModel::Optimize, py::arg("output"),
           py::arg("apply_step_damping") = false)
      .def("maybe_optimize", &AutotuneModel::MaybeOptimize, py::arg("output"));
}

}  // namespace grain::autotune
