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
"""Tests for flatmap transformation."""

import dataclasses
import itertools
import sys
from typing import Any, Sequence

from absl.testing import absltest
from grain._src.core import transforms
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import flatmap
from grain._src.python.dataset.transformations import source


@dataclasses.dataclass(frozen=True)
class FixedSizeSplitWithNoTransform(transforms.FlatMap):
  max_fan_out: int  # pyrefly: ignore[bad-override]

  def flat_map(self, element: int):
    for _ in range(self.max_fan_out):
      yield element


@dataclasses.dataclass(frozen=True)
class FixedSizeSplitWithTransform(transforms.FlatMap):
  max_fan_out: int  # pyrefly: ignore[bad-override]

  def flat_map(self, element: int):
    for _ in range(self.max_fan_out):
      yield element + 1


@dataclasses.dataclass(frozen=True)
class VariableSizeCappedSplitWithNoTransform(transforms.FlatMap):
  max_fan_out: int  # pyrefly: ignore[bad-override]

  def flat_map(self, element: int):
    return [element] * min(element, self.max_fan_out)


@dataclasses.dataclass(frozen=True)
class VariableSizeUncappedSplitWithNoTransform(transforms.FlatMap):
  max_fan_out: int  # pyrefly: ignore[bad-override]

  def flat_map(self, element: int):
    return [element] * element


@dataclasses.dataclass(frozen=True)
class SplitSentence(transforms.FlatMap):
  """Splits a sentence into its words.

  The transforms above emit indistinguishable copies of the parent element, so
  they cannot tell "split 1 of element 3" apart from "split 0 of element 3".
  Words are distinct, which is what makes a shuffle over the flattened index
  space observable.

  The fan-out is controlled by the input rather than by the transform: a
  sentence with fewer than `max_fan_out` words leaves the remaining slots as
  `None`.
  """

  max_fan_out: int  # pyrefly: ignore[bad-override]

  def flat_map(self, sentence: str) -> list[str]:
    return sentence.split()


class FlatMapMapDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = dataset.MapDataset.range(0, 10)
    self.fan_out = 10

  def test_fixed_fan_out_size(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds, FixedSizeSplitWithNoTransform(max_fan_out=self.fan_out)
    )
    self.assertLen(flatmap_ds, self.fan_out * len(self.range_ds))

  def test_flatmap_ds_length_after_repeat(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds.repeat(),
        FixedSizeSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    self.assertLen(flatmap_ds, sys.maxsize)

  def test_fixed_fan_out_data_no_transform(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds, FixedSizeSplitWithNoTransform(max_fan_out=self.fan_out)
    )
    expected_data = list(
        itertools.chain.from_iterable([[i] * self.fan_out for i in range(10)])
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_fixed_fan_out_data_with_transform(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds, FixedSizeSplitWithTransform(max_fan_out=self.fan_out)
    )
    expected_data = list(
        itertools.chain.from_iterable(
            [[i + 1] * self.fan_out for i in range(10)]
        )
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_variable_fan_out_size_still_fixed(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds,
        VariableSizeCappedSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    self.assertLen(flatmap_ds, self.fan_out * len(self.range_ds))

  def test_variable_size_fan_out_data_has_nones(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        self.range_ds,
        VariableSizeCappedSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    expected_data = list(
        itertools.chain.from_iterable(
            [[i] * i + [None] * (self.fan_out - i) for i in range(10)]
        )
    )
    actual_data = [flatmap_ds[i] for i in range(len(flatmap_ds))]
    self.assertEqual(expected_data, actual_data)

  def test_empty_dataset(self):
    flatmap_ds = flatmap.FlatMapMapDataset(
        dataset.MapDataset.range(0, 0),
        VariableSizeCappedSplitWithNoTransform(max_fan_out=self.fan_out),
    )
    self.assertEmpty(flatmap_ds)

  def test_fan_out_exceeds_max_size_raises_error(self):
    with self.assertRaises(ValueError):
      longer_range_ds = dataset.MapDataset.range(0, 20)
      flatmap_ds = flatmap.FlatMapMapDataset(
          longer_range_ds,
          VariableSizeUncappedSplitWithNoTransform(max_fan_out=self.fan_out),
      )
      _ = [flatmap_ds[i] for i in range(len(flatmap_ds))]

  def test_with_filter(self):
    ds = self.range_ds.filter(lambda x: x % 2 == 0)
    flatmap_ds = flatmap.FlatMapMapDataset(
        ds,
        FixedSizeSplitWithTransform(max_fan_out=2),
    )
    self.assertEqual(list(flatmap_ds), [1, 1, 3, 3, 5, 5, 7, 7, 9, 9])

  def test_example_docstring(self):
    @dataclasses.dataclass(frozen=True)
    class SplitWords(transforms.FlatMap):
      max_fan_out: int  # pyrefly: ignore[bad-override]

      def flat_map(self, sentence: str) -> list[str]:
        return sentence.split()

    parent_ds = dataset.MapDataset.source(["hello world", "grain"])
    self.assertEqual(parent_ds[0], "hello world")
    self.assertEqual(parent_ds[1], "grain")
    flatmap_map_ds = flatmap.FlatMapMapDataset(parent_ds, SplitWords(3))
    self.assertLen(flatmap_map_ds, 6)
    self.assertEqual(flatmap_map_ds[0], "hello")
    self.assertIsNone(flatmap_map_ds[2])
    self.assertEqual(flatmap_map_ds[3], "grain")

  def test_random_access_flatmap_then_shuffle(self):
    # Every sentence has exactly 2 words, so the fan-out is full and no slot is
    # padded with `None`.
    parent_ds = dataset.MapDataset.source(["a0 a1", "b0 b1", "c0 c1", "d0 d1"])
    flatmap_ds = flatmap.FlatMapMapDataset(parent_ds, SplitSentence(2))
    self.assertEqual(
        [flatmap_ds[i] for i in range(len(flatmap_ds))],
        ["a0", "a1", "b0", "b1", "c0", "c1", "d0", "d1"],
    )

    # The shuffle permutes the 8 flattened words rather than the 4 parent
    # sentences: the two words of "a0 a1" end up at positions 4 and 6, and
    # every word is still visited exactly once. The sentences are visited in
    # the order c, d, d, b, a, c, a, b, which is the same permutation asserted
    # by FlatMapIndexMapperTest.FlatMapThenShuffle in the C++ backend.
    shuffled_ds = flatmap_ds.shuffle(seed=42)
    self.assertEqual(
        [shuffled_ds[i] for i in range(len(shuffled_ds))],
        ["c0", "d0", "d1", "b1", "a0", "c1", "a1", "b0"],
    )

  def test_random_access_flatmap_then_shuffle_keeps_nones(self):
    # The word count of each sentence drives the fan-out, so sentences shorter
    # than `max_fan_out` leave the remaining slots as `None`.
    flatmap_ds = flatmap.FlatMapMapDataset(
        dataset.MapDataset.source(["", "b0", "c0 c1", "d0 d1 d2"]),
        SplitSentence(3),
    )
    self.assertEqual(
        [flatmap_ds[i] for i in range(len(flatmap_ds))],
        [
            None,
            None,
            None,  # "" -> no words
            "b0",
            None,
            None,  # 1 word
            "c0",
            "c1",
            None,  # 2 words
            "d0",
            "d1",
            "d2",  # 3 words
        ],
    )

    # The `None` padding is shuffled along with the real words, so the misses
    # are spread through the epoch instead of clustering after each sentence.
    shuffled_ds = flatmap_ds.shuffle(seed=42)
    self.assertEqual(
        [shuffled_ds[i] for i in range(len(shuffled_ds))],
        [
            None,
            "c0",
            "c1",
            "d0",
            None,
            None,
            None,
            None,
            None,
            "d1",
            "d2",
            "b0",
        ],
    )
    # Note that batch is only allowed on sequential datasets, so we call
    # `to_iter_dataset` first. Iterating drops the `None` padding, so only the
    # 6 real words survive and they batch evenly into 3 pairs. `batch` stacks
    # each pair into a numpy array, hence the `tolist()`.
    batched_ds = shuffled_ds.to_iter_dataset().batch(2)
    self.assertEqual(
        [batch.tolist() for batch in batched_ds],
        [["c0", "c1"], ["d0", "d1"], ["d2", "b0"]],
    )


class Unbatch(transforms.FlatMap):

  def flat_map(self, elements: Any) -> Sequence[Any]:
    return [e for e in elements]


class MyDataSource:

  def __init__(self, data: Sequence[Any]):
    self._data = data

  def __len__(self):
    return len(self._data)

  def __getitem__(self, index):
    return self._data[index]


class FlatMapIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._data = [[0, 1, 2, 3], [], [4, 5], [6, 7, 8, 9], [], [], [10, 11, 12]]
    self._data_source = MyDataSource(self._data)
    self._expected = list(range(0, 13))
    self._unbatch = Unbatch()

  def test_iter_dataset(self):
    ld = source.SourceMapDataset(self._data_source)
    iter_ds = ld.to_iter_dataset()
    iter_ds = flatmap.FlatMapIterDataset(iter_ds, self._unbatch)
    got = [e for e in iter_ds]
    self.assertSequenceEqual(got, self._expected)

  def test_checkpointing(self):
    ld = source.SourceMapDataset(self._data_source)
    iter_ds = ld.to_iter_dataset()
    iter_ds = flatmap.FlatMapIterDataset(iter_ds, self._unbatch)
    ds_iter = iter_ds.__iter__()

    max_steps = len(self._expected)
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 1, 5, 8, 9, 11, 12]:
      ds_iter.set_state(checkpoints[starting_step])
      for i in range(starting_step, max_steps):
        self.assertEqual(next(ds_iter), values_without_interruption[i])

  def test_example_docstring(self):
    class DuplicateString(transforms.FlatMap):

      def flat_map(self, element: str) -> list[str]:
        return [element, element.upper()]

    parent_ds = dataset.MapDataset.source(["hello", "grain"]).to_iter_dataset()
    flatmap_iter_ds = flatmap.FlatMapIterDataset(parent_ds, DuplicateString())
    iterator = iter(flatmap_iter_ds)
    self.assertEqual(next(iterator), "hello")
    self.assertEqual(next(iterator), "HELLO")
    self.assertEqual(next(iterator), "grain")
    self.assertEqual(next(iterator), "GRAIN")


if __name__ == "__main__":
  absltest.main()
