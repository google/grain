#ifndef THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_ARRAY_RECORD_OPS_H_
#define THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_ARRAY_RECORD_OPS_H_

#include <optional>
#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"

namespace tensorflow {
namespace data {

// Logs an error if the options_string of the written file indicates that
// group_size>1. Using larger group size can cause serious performance problems.
absl::Status CheckGroupSize(absl::string_view filename,
                            std::optional<std::string> options_string);

}  // namespace data
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_ARRAY_RECORD_OPS_H_
