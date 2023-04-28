/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "third_party/py/grain/_src/tensorflow/ops/array_record_ops.h"

#include <optional>
#include <string>
#include <tuple>

#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/googletest.h"
#include "testing/base/public/gunit.h"
#include "third_party/array_record/cpp/array_record_reader.h"
#include "third_party/array_record/cpp/array_record_writer.h"
#include "third_party/riegeli/bytes/file_reader.h"
#include "third_party/riegeli/bytes/file_writer.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::status::StatusIs;

void WriteFile(const absl::string_view filename,
               const std::optional<int> group_size = std::nullopt) {
  array_record::ArrayRecordWriterBase::Options options;
  // Default group size is 65536.
  if (group_size.has_value()) options.set_group_size(*group_size);
  array_record::ArrayRecordWriter<riegeli::FileWriter<>> writer(
      std::forward_as_tuple(filename), options);
  CHECK(writer.WriteRecord("something"));
  writer.Close();
}

TEST(ArrayRecordOpsTest, TestDefaultGroupSize) {
  const std::string filename =
      file::JoinPath(FLAGS_test_tmpdir, "file.array_record");
  WriteFile(filename);
  array_record::ArrayRecordReader<riegeli::FileReader<>> reader(
      std::forward_as_tuple(filename));
  EXPECT_THAT(CheckGroupSize(filename, reader.WriterOptionsString()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::MatchesRegex(
                           "File .* was created with group size 65536. Grain "
                           "requires group size 1 for good performance.*")));
}

TEST(ArrayRecordOpsTest, TestInvalidGroupSize) {
  const std::string filename =
      file::JoinPath(FLAGS_test_tmpdir, "file.array_record");
  WriteFile(filename, 12);
  array_record::ArrayRecordReader<riegeli::FileReader<>> reader(
      std::forward_as_tuple(filename));
  EXPECT_THAT(CheckGroupSize(filename, reader.WriterOptionsString()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::MatchesRegex(
                           "File .* was created with group size 12. Grain "
                           "requires group size 1 for good performance.*")));
}

TEST(ArrayRecordOpsTest, TestValidGroupSize) {
  const std::string filename =
      file::JoinPath(FLAGS_test_tmpdir, "file.array_record");
  array_record::ArrayRecordWriterBase::Options options;
  options.set_group_size(1);
  array_record::ArrayRecordWriter<riegeli::FileWriter<>> writer(
      std::forward_as_tuple(filename), options);
  EXPECT_TRUE(writer.WriteRecord("something"));
  writer.Close();
  array_record::ArrayRecordReader<riegeli::FileReader<>> reader(
      std::forward_as_tuple(filename));
  EXPECT_OK(CheckGroupSize(filename, reader.WriterOptionsString()));
}

TEST(ArrayRecordOpsTest, TestMissingGroupSize) {
  const std::string filename =
      file::JoinPath(FLAGS_test_tmpdir, "file.array_record");
  EXPECT_OK(CheckGroupSize(filename, std::nullopt));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
