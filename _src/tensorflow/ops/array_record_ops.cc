/* Copyright 2022 Google LLC. All Rights Reserved.

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

// TensorFlow operations for reading ArrayRecord files.
//
// ## Key vs Position
// We distinguish between the global key of a record (in a list of files) and
// the position of the record in a specific file. Each ArrayRecord file contains
// an index for its records and can do efficient reads given a list of
// positions.
// But we need to map from the global key to the file + position tuple.
// Example:
// Say we have 2 files: my_file-00000-of-00002 and my_file-00001-of-00002.
// If both files have 100 records each, then we can read keys in [0, 199].
// Key 40 will map to the record at position 40 in my_file-00000-of-00002 and
// key 121 would map to the record at position 21 in my_file-00000-of-00002.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/array_record/cpp/array_record_reader.h"
#include "third_party/array_record/cpp/thread_pool.h"
#include "third_party/re2/re2.h"
#include "third_party/riegeli/bytes/file_reader.h"
#include "third_party/tensorflow/core/framework/common_shape_fns.h"
#include "third_party/tensorflow/core/framework/op.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/op_requires.h"
#include "third_party/tensorflow/core/framework/register_types.h"
#include "third_party/tensorflow/core/framework/resource_op_kernel.h"
#include "third_party/tensorflow/core/framework/shape_inference.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/platform/errors.h"
#include "third_party/tensorflow/core/platform/mutex.h"
#include "third_party/tensorflow/core/platform/threadpool.h"
#include "third_party/tensorflow/core/platform/types.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using TfThreadPool = ::tensorflow::thread::ThreadPool;

namespace {

REGISTER_OP("ArrayRecordResourceHandle")
    .Output("handle: resource")
    .Attr("paths: list(string)")
    .Attr("cache: bool")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a ArrayRecord resource. Path must point to one or more ArrayRecord
files (shard spec or sharded file pattern are allowed).
)doc");

REGISTER_OP("ArrayRecordNumRecords")
    .Input("handle: resource")
    .Output("num_records: uint64")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Gets the number of records in the ArrayRecord resource.
)doc");

static Status ArrayRecordLookupShape(InferenceContext* c) {
  ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &input_shape));
  // Set output shape.
  c->set_output(0, input_shape);
  return OkStatus();
}

REGISTER_OP("ArrayRecordLookup")
    .Input("handle: resource")
    .Input("keys: dtype")
    .Output("records: string")
    .Attr("dtype: {int32, uint32, int64, uint64}")
    .SetShapeFn(ArrayRecordLookupShape)
    .Doc(R"doc(
Reads a list of keys from the ArrayRecord resource.
)doc");

// Holds a filename and a range [start, end) to read.
// The file needs to contain at least `end` records.
struct ReadInstruction {
  const std::string filename;
  const uint64_t start;
  const uint64_t end;

  uint64_t NumRecords() const { return end - start; }
};

Status GetReadInstructions(const std::string& path,
                           std::vector<ReadInstruction>& read_instructions) {
  const string pattern = path;
  // Find all matching filenames.
  const auto status_or_filenames = file::Match(pattern, file::Defaults());
  if (!status_or_filenames.ok() || status_or_filenames.value().empty()) {
    return Status(
        error::NOT_FOUND,
        absl::StrFormat("Failed to find matching files pattern %s.", pattern));
  }
  auto filenames = status_or_filenames.value();
  // Make sure we always read files in the same order.
  absl::c_sort(filenames);
  if (filenames.size() > 100) {
    LOG(WARNING) << "Constructing a global index for over 100 files can be "
                 << "slow. Consider providing read instruction strings "
                 << "('filename[start:end]') to ArrayRecord to avoid this.";
  }
  for (const std::string& filename : filenames) {
    if (!file::Exists(filename, file::Defaults()).ok()) {
      return Status(error::NOT_FOUND,
                    absl::StrFormat("File %s not found.", filename));
    }
    const array_record::ArrayRecordReader<riegeli::FileReader<>> reader(
        std::forward_as_tuple(filename));
    read_instructions.push_back({filename, 0, reader.NumRecords()});
  }
  return OkStatus();
}

// Threadsafe cache for values of type tstring.
class RecordCache {
 public:
  bool Lookup(const uint64_t key, tstring& value) {
    std::lock_guard<mutex> lock(mu_);
    auto entry = cache_.find(key);
    if (entry == cache_.end()) {
      return false;
    }
    value = entry->second;
    return true;
  }

  void Insert(const uint64_t key, absl::string_view value) {
    std::lock_guard<mutex> lock(mu_);
    const auto [it, inserted] = cache_.insert({key, tstring(value)});
    if (inserted) {
      num_cached_bytes_ += it->second.size() * sizeof(char);
      const int64_t cache_memory_usage =
          cache_.bucket_count() * sizeof(std::pair<uint64_t, tstring>) +
          num_cached_bytes_;
      LOG_EVERY_N_SEC(INFO, 60)
          << "ArrayRecordResource cache uses " << cache_memory_usage
          << " bytes of memory for " << cache_.size() << " elements.";
    } else {
      // Element already in cache. This can happen if multiple threads requested
      // `key` before it was in the cache.
      if (it->second == value) {
        LOG(WARNING) << "Element with key " << key << " already in cache.";
      } else {
        LOG(FATAL) << "Element with key " << key
                   << " already in cache but value changed.\nOld value: "
                   << it->second << "\nNew value: " << value;
      }
    }
  }

 private:
  absl::flat_hash_map<uint64_t, tstring> cache_;
  int64_t num_cached_bytes_ = 0;
  mutex mu_;
};

// Resource that holds the file reader objects and implements the lookup logic.
// Init() constructs the global index by reading the number of records per file.
// NumRecords() returns the total number of records.
// Lookup() looks up the provided keys and returns the records. If needed it
// will open file readers.
class ArrayRecordResource : public ResourceBase {
 public:
  explicit ArrayRecordResource(const std::vector<string>& path, bool use_cache)
      : ResourceBase(), paths_(path), use_cache_(use_cache) {}

  Status Init() {
    RE2 pattern(R"((.+)\[(\d+):(\d+)\])");
    for (const string& path : paths_) {
      std::string filename;
      uint64_t start, end;
      if (RE2::FullMatch(path, pattern, &filename, &start, &end)) {
        read_instructions_.push_back({filename, start, end});
      } else {
        TF_RETURN_IF_ERROR(GetReadInstructions(path, read_instructions_));
      }
    }
    total_num_records_ = 0;
    for (const auto& ri : read_instructions_) {
      total_num_records_ += ri.NumRecords();
    }
    readers_.resize(read_instructions_.size());
    return OkStatus();
  }

  string DebugString() const override {
    return strings::StrCat("ArrayRecordResource(", absl::StrJoin(paths_, ","),
                           ")");
  }

  uint64_t NumRecords() const { return total_num_records_; }

  Status Lookup(OpKernelContext* context, const std::vector<uint64_t> keys,
                absl::Span<tstring> records) {
    // Read records pointed to by keys and put them into records.
    // **key** is the global key of a record over all files.
    // **position** is the position of a record in a specific file/reader.
    // **idx** is the in the keys vector. records should be filled with the
    // corresponding values (in the same order).

    // Get the positions and the indices each reader should read.
    // There is one reader per file.
    std::vector<std::vector<uint64_t>> indices_per_reader;
    std::vector<std::vector<uint64_t>> positions_per_reader;
    indices_per_reader.resize(readers_.size());
    positions_per_reader.resize(readers_.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (use_cache_ && cache_.Lookup(keys[i], records[i])) {
        continue;
      }
      int reader_index;
      uint64_t position;
      std::tie(reader_index, position) = GetReaderIndexAndPosition(keys[i]);
      indices_per_reader[reader_index].push_back(i);
      positions_per_reader[reader_index].push_back(position);
    }

    // Only perform lookup for readers that have reads scheduled.
    // Each reader will use a separate thread.
    std::vector<int> readers_with_reads;
    for (int i = 0; i < readers_.size(); ++i) {
      if (!positions_per_reader[i].empty()) {
        readers_with_reads.push_back(i);
      }
    }

    auto perform_lookups = [&](int start, int end) {
      for (int i = start; i < end; ++i) {
        const int reader_index = readers_with_reads[i];
        CHECK(!positions_per_reader[reader_index].empty())
            << "No positions for reader " << reader_index;
        if (readers_[reader_index] == nullptr) {
          CreateReader(reader_index);
        }
        const auto status =
            readers_[reader_index]->ParallelReadRecordsWithIndices(
                positions_per_reader[reader_index],
                [&](uint64_t read_idx,
                    absl::string_view record) -> absl::Status {
                  const size_t idx = indices_per_reader[reader_index][read_idx];
                  if (use_cache_) {
                    cache_.Insert(keys[idx], record);
                  }
                  records[idx] = record;
                  return absl::OkStatus();
                });
        if (!status.ok()) {
          LOG(ERROR) << "Failed to read from reader " << reader_index
                     << " (filename="
                     << read_instructions_[reader_index].filename << ")"
                     << ": " << status;
        }
      }
    };

    TfThreadPool::SchedulingParams scheduling_params(
        TfThreadPool::SchedulingStrategy::kFixedBlockSize, std::nullopt,
        /* block_size= */ 1);
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        readers_with_reads.size(), scheduling_params, perform_lookups);

    return OkStatus();
  }

 private:
  using Reader =
      std::unique_ptr<array_record::ArrayRecordReader<riegeli::FileReader<>>>;

  const std::vector<string> paths_;
  std::vector<ReadInstruction> read_instructions_;
  uint64_t total_num_records_;

  const bool use_cache_;
  RecordCache cache_;

  std::vector<Reader> readers_;
  mutex create_reader_mutex_;

  std::pair<int, uint64_t> GetReaderIndexAndPosition(uint64_t key) const {
    int reader_index = 0;
    CHECK(key < NumRecords()) << "Invalid key " << key;
    while (key >= read_instructions_[reader_index].NumRecords()) {
      key -= read_instructions_[reader_index].NumRecords();
      reader_index++;
    }
    key += read_instructions_[reader_index].start;
    return {reader_index, key};
  }

  void CreateReader(const int reader_index) {
    const std::lock_guard<mutex> lock(create_reader_mutex_);
    if (readers_[reader_index] == nullptr) {
      riegeli::FileReaderBase::Options file_reader_options;
      // Set buffer size to 32 KiB. The default of 1 MiB doesn't work well for
      // random access pattern when individual records are small (<= 100 KiB).
      file_reader_options.set_buffer_size(1 << 15);
      // Copy is on purpose.
      std::string filename = read_instructions_[reader_index].filename;
      readers_[reader_index] = std::make_unique<
          array_record::ArrayRecordReader<riegeli::FileReader<>>>(
          std::forward_as_tuple(filename, file_reader_options),
          array_record::ArrayRecordReaderBase::Options(),
          array_record::ArrayRecordGlobalPool());
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ArrayRecordResource);
};

// TF op for creating the ArrayRecordResource and returning a handle for it.
class ArrayRecordResourceHandleOp
    : public ResourceOpKernel<ArrayRecordResource> {
 public:
  explicit ArrayRecordResourceHandleOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("paths", &paths_));
    OP_REQUIRES_OK(context, context->GetAttr("cache", &use_cache_));
  }

 private:
  Status CreateResource(ArrayRecordResource** ret) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    *ret = new ArrayRecordResource(paths_, use_cache_);
    return (*ret)->Init();
  }

  std::vector<string> paths_;
  bool use_cache_;

  TF_DISALLOW_COPY_AND_ASSIGN(ArrayRecordResourceHandleOp);
};

REGISTER_KERNEL_BUILDER(Name("ArrayRecordResourceHandle").Device(DEVICE_CPU),
                        ArrayRecordResourceHandleOp);

class ArrayRecordNumRecordsOp : public OpKernel {
 public:
  explicit ArrayRecordNumRecordsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ArrayRecordResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));
    core::ScopedUnref scoped_unref(resource);

    Tensor* num_records_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &num_records_t));
    num_records_t->scalar<uint64_t>()() = resource->NumRecords();
  }
};

REGISTER_KERNEL_BUILDER(Name("ArrayRecordNumRecords").Device(DEVICE_CPU),
                        ArrayRecordNumRecordsOp);

// Lookup op that uses the ArrayRecordResource to map a tensor of keys to a
// tensor of records.
template <typename KeyType>
class ArrayRecordLookupOp : public OpKernel {
 public:
  explicit ArrayRecordLookupOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    ArrayRecordResource* resource;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0),
                                           &resource));
    core::ScopedUnref scoped_unref(resource);

    const Tensor keys_t = context->input(1);
    const TensorShape& shape = keys_t.shape();
    const int num_keys = shape.num_elements();

    std::vector<uint64_t> keys;
    keys.reserve(num_keys);

    for (int i = 0; i < num_keys; ++i) {
      const uint64_t key = static_cast<uint64_t>(keys_t.flat<KeyType>()(i));
      keys.push_back(key);
    }
    Tensor* records_t;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &records_t));
    auto records_flat = records_t->flat<tstring>();
    absl::Span<tstring> records(records_flat.data(), records_flat.size());

    OP_REQUIRES_OK(context, resource->Lookup(context, keys, records));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ArrayRecordLookupOp);
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("ArrayRecordLookup")           \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          ArrayRecordLookupOp<TYPE>);

TF_CALL_int32(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_uint32(REGISTER);
TF_CALL_uint64(REGISTER);

}  // namespace
}  // namespace tensorflow
