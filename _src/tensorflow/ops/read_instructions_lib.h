#ifndef THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_READ_INSTRUCTIONS_LIB_H_
#define THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_READ_INSTRUCTIONS_LIB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

struct ReadInstruction {
  std::string filename;
  int64_t start = 0;  // Always >= 0.
  // Must be >= start or -1. -1 indicates that the end of the file.
  int64_t end = -1;

  static tsl::StatusOr<ReadInstruction> Parse(absl::string_view path);

  int64_t NumRecords() const { return end - start; }
};

using GetNumRecords = std::function<uint64_t(const std::string&)>;

// Get the read instructions for a list of paths where each path can be:
// - A normal filename.
// - A filename with read instructions: filename[start:end].
// Unless the filename is given with read instruction the file will be opened
// to get the total number of records.
tsl::StatusOr<std::vector<ReadInstruction>> GetReadInstructions(
    const std::vector<std::string>& paths,
    const GetNumRecords& get_num_records);

}  // namespace data
}  // namespace tensorflow

#endif  // THIRD_PARTY_PY_GRAIN__SRC_TENSORFLOW_OPS_READ_INSTRUCTIONS_LIB_H_
