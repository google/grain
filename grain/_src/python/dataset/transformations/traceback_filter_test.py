# Copyright 2025 Google LLC
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

import traceback
from typing import Callable

from grain._src.core import traceback_util
import grain.python as pygrain

from absl.testing import absltest


traceback_util.register_exclusion(__file__)


def _assert_exception_with_short_traceback(
    self: absltest.TestCase,
    fn: Callable[[], None],
    expected_error_type: type[BaseException],
) -> None:
  """Asserts that a function raises a specific exception with a short traceback.

  This function executes `fn` and asserts that it raises an exception of type
  `expected_error_type`. Additionally, it checks that the length of the
  traceback associated with the caught exception is less than 15 frames,
  ensuring that the traceback has been shortened.

  Args:
    self: The absltest.TestCase instance.
    fn: The function to execute, expected to raise an exception.
    expected_error_type: The expected type of the exception to be raised.
  """
  # Assert that an exception is raised and the length of the traceback is
  # sufficiently short, i.e. less than 15 frames.
  # We cannot use assertRaises because __traceback__ is cleared before we can
  # inspect it.
  try:
    fn()
    self.fail(f"Expected {expected_error_type} to be raised.")
  except expected_error_type as e:
    tb = traceback.extract_tb(e.__traceback__)
  except Exception as e:  # pylint: disable=broad-except
    self.fail(
        f"Expected {expected_error_type} to be raised, but got"
        f" {type(e)} instead."
    )
  self.assertLess(len(tb), 15, f"Traceback is too long: \n{tb}")


class AddOneTransform(pygrain.MapTransform):

  def map(self, x: int) -> int:
    return x + 1


class RaiseErrorTransform(pygrain.MapTransform):

  def map(self, x: int) -> int:
    raise ValueError("Boom!")


class TracebackFilterTest(absltest.TestCase):

  def test_datasource_multiple_transforms_filters_traceback(self):
    range_ds = pygrain.RangeDataSource(0, 10, 1)
    sampler = pygrain.IndexSampler(num_records=10, seed=42)
    ops = [RaiseErrorTransform()]
    for _ in range(100):
      ops.append(AddOneTransform())
    data_loader = pygrain.DataLoader(
        data_source=range_ds, sampler=sampler, operations=ops
    )
    _assert_exception_with_short_traceback(
        self, lambda: next(iter(data_loader)), ValueError
    )

  def test_dataset_multiple_transforms_filters_traceback(self):
    range_ds = pygrain.MapDataset.range(0, 10)
    range_ds = range_ds.map(RaiseErrorTransform())
    for _ in range(100):
      range_ds = range_ds.map(AddOneTransform())
    _assert_exception_with_short_traceback(
        self, lambda: next(iter(range_ds)), ValueError
    )


if __name__ == "__main__":
  absltest.main()
