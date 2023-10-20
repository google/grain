# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from grain._src.python.experimental.proto_parsers import fast_proto_parser
import numpy as np
from absl.testing import absltest
from tensorflow.core.example import example_pb2


class FastProtoParserTest(absltest.TestCase):

  def test_parse_example_containing_int64(self):
    expected_array = np.arange(1, 1_000_000, dtype=np.int64)
    e = example_pb2.Example()
    e.features.feature["foo"].int64_list.value.extend(expected_array)

    e_parsed = fast_proto_parser.parse_tf_example(e.SerializeToString())

    self.assertEqual(list(e_parsed.keys()), ["foo"])
    self.assertTrue(
        np.array_equal(e_parsed["foo"], expected_array),
        f"{e_parsed['foo']=}, {expected_array=}",
    )

  def test_parse_example_containing_float(self):
    expected_array = np.arange(0.1, 100_000.0, 0.1, dtype=np.float32)
    e = example_pb2.Example()
    e.features.feature["bar"].float_list.value.extend(expected_array)

    e_parsed = fast_proto_parser.parse_tf_example(e.SerializeToString())

    self.assertEqual(list(e_parsed.keys()), ["bar"])
    self.assertTrue(
        np.array_equal(e_parsed["bar"], expected_array),
        f"{e_parsed['bar']=}, {expected_array=}",
    )

  def test_parse_example_containing_bytes(self):
    expected_array = np.array([b"abc", b"cde123", b"bazzzzzz"] * 100_000)
    e = example_pb2.Example()
    e.features.feature["baz"].bytes_list.value.extend(expected_array)

    e_parsed = fast_proto_parser.parse_tf_example(e.SerializeToString())

    self.assertEqual(list(e_parsed.keys()), ["baz"])
    self.assertTrue(
        np.array_equal(e_parsed["baz"], expected_array),
        f"{e_parsed['baz']=}, {expected_array=}",
    )

  def test_parse_example_containing_all_types(self):
    expected_output = {
        "foo": np.arange(1, 1_000_000.0, dtype=np.int64),
        "bar": np.arange(0.1, 100_000.0, 0.1, dtype=np.float32),
        "baz": np.array([b"abc", b"cde123", b"bazzzzzz"] * 100_000),
    }
    e = example_pb2.Example()
    e.features.feature["foo"].int64_list.value.extend(expected_output["foo"])
    e.features.feature["bar"].float_list.value.extend(expected_output["bar"])
    e.features.feature["baz"].bytes_list.value.extend(expected_output["baz"])

    e_parsed = fast_proto_parser.parse_tf_example(e.SerializeToString())

    self.assertEqual(e_parsed.keys(), expected_output.keys())
    for feature_name in e_parsed.keys():
      self.assertTrue(
          np.array_equal(e_parsed[feature_name], expected_output[feature_name]),
          f"{e_parsed[feature_name]=}, {expected_output[feature_name]=}",
      )

  def test_parse_example_containing_empty_list(self):
    e = example_pb2.Example()
    e.features.feature["foo"].int64_list.value[:] = []

    e_parsed = fast_proto_parser.parse_tf_example(e.SerializeToString())

    self.assertEqual(list(e_parsed.keys()), ["foo"])
    self.assertEmpty(e_parsed["foo"])


if __name__ == "__main__":
  absltest.main()
