#include "third_party/py/grain/_src/tensorflow/ops/read_instructions_lib.h"

#include "third_party/array_record/cpp/array_record_reader.h"
#include "third_party/re2/re2.h"
#include "third_party/riegeli/bytes/file_reader.h"
#include "third_party/tensorflow/core/platform/errors.h"
#include "third_party/tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace data {

// Getting the read instructions is cheap but IO bound. We create a temporary
// thread pool to get the number of records.
constexpr int kNumThreadsForReadInstructions = 256;

using TfThreadPool = ::tensorflow::thread::ThreadPool;

tsl::StatusOr<ReadInstruction> ReadInstruction::Parse(absl::string_view path) {
  static const LazyRE2 kPattern = {R"((.+)\[(\d+):(\d+)\])"};
  std::string filename;
  int64_t start, end;
  if (RE2::FullMatch(path, *kPattern, &filename, &start, &end)) {
    return ReadInstruction{filename, start, end};
  }
  return errors::InvalidArgument("Can't parse ", path, " as ReadInstruction");
}

// Get the read instructions for a list of paths where each path can be:
// - A normal filename.
// - A filename with read instructions: filename[start:end].
// Unless the filename is given with read instruction, the file will be opened
// to get the total number of records.
tsl::StatusOr<std::vector<ReadInstruction>> GetReadInstructions(
    const std::vector<std::string>& paths,
    const GetNumRecords& get_num_records) {
  std::vector<ReadInstruction> read_instructions;

  // Step 1: Parse potential read instructions.
  bool missing_num_records = false;
  for (const std::string& path : paths) {
    tsl::StatusOr<ReadInstruction> read_instruction =
        ReadInstruction::Parse(path);
    if (read_instruction.ok()) {
      read_instructions.push_back(read_instruction.value());
    } else {
      missing_num_records = true;
      const std::string pattern = path;
      read_instructions.push_back({pattern});
    }
  }
  if (!missing_num_records) {
    return read_instructions;
  }

  TfThreadPool thread_pool(Env::Default(), "get_read_instructions",
                           kNumThreadsForReadInstructions);
  const TfThreadPool::SchedulingParams scheduling_params(
      TfThreadPool::SchedulingStrategy::kFixedBlockSize,
      /*cost_per_unit=*/std::nullopt,
      /* block_size= */ 1);

  std::vector<std::vector<ReadInstruction>> filled_instructions;
  filled_instructions.resize(read_instructions.size());

  // Step 2: Match any patterns.
  auto match_pattern = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      const std::string& pattern = read_instructions[i].filename;
      if (read_instructions[i].end >= 0 || !absl::StrContains(pattern, '?')) {
        filled_instructions[i].push_back(std::move(read_instructions[i]));
        continue;
      }
      const auto status_or_filenames = file::Match(pattern, file::Defaults());
      if (!status_or_filenames.ok() || status_or_filenames->empty()) {
        LOG(ERROR) << "Failed to find matching files for pattern " << pattern;
        continue;
      }
      auto filenames = *status_or_filenames;
      // Make sure we always read files in the same order.
      absl::c_sort(filenames);
      filled_instructions[i].reserve(filenames.size());
      for (const std::string& filename : filenames) {
        filled_instructions[i].push_back({filename, 0, -1});
      }
    }
  };

  thread_pool.ParallelFor(read_instructions.size(), scheduling_params,
                          match_pattern);

  // Flatten filled_instructions into read_instructions;
  read_instructions.clear();
  for (const auto& instructions : filled_instructions) {
    read_instructions.insert(read_instructions.end(), instructions.begin(),
                             instructions.end());
  }

  // Step 3: Get number of records.
  auto add_num_records = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      if (read_instructions[i].end >= 0) {
        continue;
      }
      const std::string& filename = read_instructions[i].filename;
      if (!file::Exists(filename, file::Defaults()).ok()) {
        LOG(ERROR) << "File " << filename << " not found.";
        continue;
      }
      read_instructions[i].end =
          static_cast<int64_t>(get_num_records(filename));
    }
  };
  thread_pool.ParallelFor(read_instructions.size(), scheduling_params,
                          add_num_records);
  return read_instructions;
}

}  // namespace data
}  // namespace tensorflow
