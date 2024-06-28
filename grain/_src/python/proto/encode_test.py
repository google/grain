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

import dataclasses
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.proto import encode
import jax.numpy as jnp
import numpy as np


class MakeTfExampleTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_dataclasses(self, fast_bfloat16: bool):
    @dataclasses.dataclass(frozen=True)
    class Holder:
      ints: List[int]
      floats: List[float]
      bytez: List[bytes]
      bools: List[bool]
      bfloat16s: jnp.ndarray

    data = Holder(
        ints=[1, 2, 3],
        floats=[1.0, 2.0, 3.0],
        bytez=[b"\0", b"\1", b"\2"],
        bools=[False, True, True],
        bfloat16s=jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.bfloat16),
    )
    features = dataclasses.asdict(data)
    example = encode.make_tf_example(features, fast_bfloat16=fast_bfloat16)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key in features:
      type_map = {
          "ints": "int64_list",
          "floats": "float_list",
          "bytez": "bytes_list",
          "bools": "int64_list",
          "bfloat16s": "bytes_list",
      }
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"),
          type_map[key],
          f"wrong type for {key}",
      )

  def test_bool_dtype(self):
    rng = np.random.default_rng(333)
    features = {
        "bool": rng.integers(0, high=2, size=(100,), dtype=np.int32) == 2,
    }
    example = encode.make_tf_example(features)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      # Bools are commonly stored as int64 features. Probably no one made the
      # effort to store them as bits. At least protobuf uses variable length
      # integers.
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"), "int64_list"
      )
      self.assertSequenceEqual(
          [bool(x) for x in example.features.feature[key].int64_list.value],
          list(expected),
      )

  def test_int_dtypes(self):
    rng = np.random.default_rng(333)
    features = {
        "int8": rng.integers(low=-100, high=100, size=(1000,), dtype=np.int8),
        "uint8": rng.integers(low=0, high=100, size=(1000,), dtype=np.uint8),
        "int16": rng.integers(low=-100, high=100, size=(1000,), dtype=np.int16),
        "uint16": rng.integers(low=0, high=100, size=(1000,), dtype=np.uint16),
        "int32": rng.integers(low=-100, high=100, size=(1000,), dtype=np.int32),
        "uint32": rng.integers(low=0, high=100, size=(1000,), dtype=np.uint32),
        "int64": rng.integers(2**31, high=2**33, size=(1000,), dtype=np.int64),
        "uint64": rng.integers(2**31, 2**33, size=(1090,), dtype=np.uint64),
        "list_int": list(rng.integers(2**31, high=2**33, size=(1000,))),
        "tuple_int": tuple(rng.integers(2**31, high=2**33, size=(1000,))),
    }
    example = encode.make_tf_example(features)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"),
          "int64_list",
          msg=(
              f"{key} had kind"
              f" {example.features.feature[key].WhichOneof('kind')}"
          ),
      )
      np.testing.assert_array_equal(
          example.features.feature[key].int64_list.value, expected
      )

  def test_float_dtypes(self):
    rng = np.random.default_rng(333)
    features = {
        "float32": rng.random(size=(100,), dtype=np.float32),
        "float64": rng.random(size=(100,), dtype=np.float64),
        "list_float32": list(rng.random(size=(100,), dtype=np.float32)),
    }
    example = encode.make_tf_example(features)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"), "float_list"
      )
      np.testing.assert_allclose(
          example.features.feature[key].float_list.value, expected
      )

  def test_bytes_dtypes(self):
    features = {
        "bytes_array": np.asarray(b"12345"),
        "bytes": b"bla",
        "string_encoded": "uiae".encode("utf-8"),
        "bytes_list": [b"bla", b"foo"],
    }
    example = encode.make_tf_example(features)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"), "bytes_list"
      )
      np.testing.assert_array_equal(
          example.features.feature[key].bytes_list.value, expected
      )

  @parameterized.parameters([True, False])
  def test_bfloat16_dtype(self, fast_bfloat16: bool):
    x = jnp.asarray([0.23, 1.2, 4.448], dtype=jnp.bfloat16)
    features = {
        "jnp_bfloat16": x,
        "np_bfloat16": np.asarray(x),
    }
    example = encode.make_tf_example(features, fast_bfloat16=fast_bfloat16)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      # bfloat16 features are stored as bytes.
      if fast_bfloat16:
        expected = [expected.tobytes()]
      else:
        expected = [x.tobytes() for x in expected]
      self.assertEqual(example.features.feature[key].bytes_list.value, expected)

  def test_int_2d(self):
    rng = np.random.default_rng(333)
    features = {
        "int32": rng.integers(low=-100, high=100, size=(5, 10), dtype=np.int32),
        "int64": rng.integers(
            2**31,
            high=2**33,
            size=(
                10,
                5,
            ),
            dtype=np.int64,
        ),
    }
    example = encode.make_tf_example(features)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      self.assertEqual(
          example.features.feature[key].WhichOneof("kind"),
          "int64_list",
          msg=(
              f"{key} had kind"
              f" {example.features.feature[key].WhichOneof('kind')}"
          ),
      )
      np.testing.assert_array_equal(
          example.features.feature[key].int64_list.value, expected.flatten()
      )

  @parameterized.parameters([True, False])
  def test_bfloat16_2d(self, fast_bfloat16: bool):
    x = jnp.asarray(
        [[1.23, 2.2, 4.448], [2.23, 4.2, 8.448]], dtype=jnp.bfloat16
    )
    features = {
        "jnp_bfloat16": x,
        "np_bfloat16": np.asarray(x),
    }
    example = encode.make_tf_example(features, fast_bfloat16=fast_bfloat16)
    self.assertSetEqual(set(example.features.feature.keys()), set(features))
    for key, expected in features.items():
      # bfloat16 features are stored as bytes.
      if fast_bfloat16:
        expected = [expected.tobytes()]
      else:
        expected = [x.tobytes() for x in expected.flatten()]
      np.testing.assert_array_equal(
          example.features.feature[key].bytes_list.value, expected
      )

  def test_fortan_contiguous_fails(self):
    features = {"a": np.asarray([[1, 2, 3], [4, 5, 6]], order="F")}
    with self.assertRaises(ValueError):
      encode.make_tf_example(features)


if __name__ == "__main__":
  absltest.main()
