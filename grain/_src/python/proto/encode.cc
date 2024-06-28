/* Copyright 2023 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdint>
#include <string>
#include <utility>

#include "third_party/absl/strings/str_cat.h"
#include "third_party/protobuf/map.h"
#include "third_party/protobuf/repeated_field.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "third_party/py/numpy/core/include/numpy/ndarraytypes.h"
#include "third_party/pybind11/include/pybind11/cast.h"
#include "third_party/pybind11/include/pybind11/detail/common.h"
#include "third_party/pybind11/include/pybind11/gil.h"
#include "third_party/pybind11/include/pybind11/numpy.h"
#include "third_party/pybind11/include/pybind11/pybind11.h"
#include "third_party/pybind11/include/pybind11/pytypes.h"
#include "third_party/pybind11/include/pybind11/stl.h"  // IWYU pragma: keep (necessary for py::cast from std::vector)
#include "third_party/pybind11_protobuf/native_proto_caster.h"
#include "third_party/tensorflow/core/example/example.proto.h"
#include "third_party/tensorflow/core/example/feature.proto.h"

namespace py = ::pybind11;

namespace {

template <typename DType>
::tensorflow::Int64List AsIntList(py::array np_array) {
  ::tensorflow::Int64List int64_list;
  const DType* data = static_cast<const DType*>(np_array.data());
  int64_list.mutable_value()->Assign(data, data + np_array.size());
  return int64_list;
}

template <typename DType>
::tensorflow::FloatList AsFloatList(py::array np_array) {
  ::tensorflow::FloatList float_list;
  const DType* data = static_cast<const DType*>(np_array.data());
  float_list.mutable_value()->Assign(data, data + np_array.size());
  return float_list;
}

::tensorflow::BytesList AsBytesList(py::array np_array) {
  ::tensorflow::BytesList bytes_list;
  if (np_array.ndim() == 0) {
    const char* data = static_cast<const char*>(np_array.data());
    bytes_list.add_value(data, np_array.itemsize());
  } else {
    bytes_list.mutable_value()->Reserve(np_array.size());
    np_array = np_array.reshape({np_array.size()});
    for (int i = 0; i < np_array.size(); i++) {
      const char* data = static_cast<const char*>(np_array.data(i));
      bytes_list.add_value(data, np_array.itemsize());
    }
  }
  return bytes_list;
}

bool ContainsElementsConvertibleToBfloat16s(py::array np_array) {
  return np_array.itemsize() == 2;
}

::tensorflow::BytesList Bfloat16sAsPackedBytesList(py::array np_array) {
  ::tensorflow::BytesList bytes_list;
  const char* data = static_cast<const char*>(np_array.data());
  bytes_list.add_value(data, np_array.nbytes());
  return bytes_list;
}

}  // namespace

PYBIND11_MODULE(encode, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() = "Encode array-like NumPy arrays to `tensorflow.Example` protobuf.";

  // Passing `fast_bfloat16` == true packs NumPy arrays of an unrecognized type
  // (e.g., pure bytes) having elements of size equal to 2 bytes into one large
  // blob of data.
  m.def(
      "make_tf_example",
      [](py::dict features, bool fast_bfloat16) -> tensorflow::Example {
        tensorflow::Example example;
        proto2::Map<std::string, tensorflow::Feature>& feature_map =
            *example.mutable_features()->mutable_feature();
        for (auto& [feature_name, value] : features) {
          py::array np_array = value.cast<py::array>();
          if (!(np_array.flags() & py::array::c_style)) {
            throw py::value_error(absl::StrCat(
                "Only c_style arrays are supported but features ",
                feature_name.cast<std::string>(), " is not c_style."));
          }
          auto& feature = feature_map[feature_name.cast<std::string>()];

          // Floating point types.
          if (py::isinstance<py::array_t<float>>(np_array)) {
            *feature.mutable_float_list() = AsFloatList<float>(np_array);
          } else if (py::isinstance<py::array_t<double>>(np_array)) {
            *feature.mutable_float_list() = AsFloatList<double>(np_array);
            // Integer types.
          } else if (py::isinstance<py::array_t<int8_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<int8_t>(np_array);
          } else if (py::isinstance<py::array_t<uint8_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<uint8_t>(np_array);
          } else if (py::isinstance<py::array_t<int16_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<int16_t>(np_array);
          } else if (py::isinstance<py::array_t<uint16_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<uint16_t>(np_array);
          } else if (py::isinstance<py::array_t<int32_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<int32_t>(np_array);
          } else if (py::isinstance<py::array_t<uint32_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<uint32_t>(np_array);
          } else if (py::isinstance<py::array_t<int64_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<int64_t>(np_array);
          } else if (py::isinstance<py::array_t<uint64_t>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<uint64_t>(np_array);

            // Bools are commonly stored as int64 features.
          } else if (py::isinstance<py::array_t<bool>>(np_array)) {
            *feature.mutable_int64_list() = AsIntList<bool>(np_array);

            // String types.
          } else if (np_array.dtype().num() == NPY_TYPES::NPY_UNICODE) {
            // Unicode string.
            throw py::value_error(
                absl::StrCat("Feature ", feature_name.cast<std::string>(),
                             " is a unicode string. This is not supported, "
                             "please pass bytes by calling `.encode('utf-8')` "
                             "on the feature."));
          } else if (np_array.dtype().num() == NPY_TYPES::NPY_STRING) {
            // Byte string.
            *feature.mutable_bytes_list() = AsBytesList(np_array);

            // Everything else simply gets stored as bytes.
            // This includes bfloat16 values which don't have a corresponding
            // proto type.
          } else {
            if (fast_bfloat16 &&
                ContainsElementsConvertibleToBfloat16s(np_array)) {
              *feature.mutable_bytes_list() =
                  Bfloat16sAsPackedBytesList(np_array);
            } else {
              *feature.mutable_bytes_list() = AsBytesList(np_array);
            }
          }
        }
        return example;
      },
      py::arg("features"), py::kw_only(), py::arg("fast_bfloat16") = false,
      py::return_value_policy::take_ownership);
}
