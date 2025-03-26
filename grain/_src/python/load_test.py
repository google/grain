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
"""Tests for load()."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
import multiprocessing as mp
from grain._src.python import data_sources
from grain._src.python import load
import numpy as np


FLAGS = flags.FLAGS


class FilterEven(transforms.Filter):

  def filter(self, x: int) -> bool:
    return x % 2 == 0


class PlusOne(transforms.MapTransform):

  def map(self, x: int) -> int:
    return x + 1


class DataLoaderTest(parameterized.TestCase):

  def test_with_range_source(self):
    range_data_source = data_sources.RangeDataSource(start=0, stop=8, step=1)
    transformations = [
        PlusOne(),
        FilterEven(),
    ]
    data_loader = load.load(
        range_data_source, transformations=transformations, num_epochs=1
    )
    expected = [2, 4, 6, 8]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)

  def test_with_range_source_with_batch(self):
    range_data_source = data_sources.RangeDataSource(start=0, stop=8, step=1)
    transformations = [
        PlusOne(),
        FilterEven(),
    ]
    data_loader = load.load(
        range_data_source,
        transformations=transformations,
        batch_size=2,
        num_epochs=1,
    )
    expected = [np.array([2, 4]), np.array([6, 8])]
    actual = list(data_loader)
    np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()
