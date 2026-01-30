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
"""Tests for traceback filtering utilities."""

import pathlib
import traceback
from typing import Callable
from unittest import mock

from absl.testing import absltest
import grain
from grain._src.core import traceback_util
from grain._src.core.config import config  # pylint: disable=g-importing-member


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


@traceback_util.run_with_traceback_filter
def raise_after_recursion(curr_count: int, max_count: int) -> None:
  if curr_count == max_count:
    raise ValueError("Boom!")
  raise_after_recursion(curr_count + 1, max_count)


class TracebackUtilTest(absltest.TestCase):

  def test_traceback_mode_auto_decorator_filters_traceback(self):
    config.update("py_traceback_filtering", "auto")
    _assert_exception_with_short_traceback(
        self, lambda: raise_after_recursion(0, 150), ValueError
    )

  def test_traceback_mode_remove_frames_decorator_filters_traceback(self):
    config.update("py_traceback_filtering", "remove_frames")
    _assert_exception_with_short_traceback(
        self, lambda: raise_after_recursion(0, 150), ValueError
    )

  def test_traceback_mode_quiet_remove_frames_decorator_filters_traceback(self):
    config.update("py_traceback_filtering", "quiet_remove_frames")
    _assert_exception_with_short_traceback(
        self, lambda: raise_after_recursion(0, 150), ValueError
    )

  def test_traceback_mode_off_decorator_does_not_filter_traceback(self):
    config.update("py_traceback_filtering", "off")
    try:
      raise_after_recursion(0, 150)
      self.fail("Expected ValueError to be raised.")
    except ValueError as e:
      tb = traceback.extract_tb(e.__traceback__)
    self.assertGreater(len(tb), 150)

  def test_traceback_mode_tracebackhide_decorator_applies_tracebackhide(self):
    config.update("py_traceback_filtering", "tracebackhide")
    frame_locals = []
    try:
      raise_after_recursion(0, 150)
      self.fail("Expected ValueError to be raised.")
    except ValueError as e:
      tb = traceback.walk_tb(e.__traceback__)
      for f, _ in tb:
        frame_locals.append((f.f_code.co_filename, f.f_locals))
    frame_count = 0
    # Verify expected frames are marked with __tracebackhide__.
    for index, filename_local_vars in enumerate(frame_locals):
      filename, local_vars = filename_local_vars
      frame_count += 1
      if index == 0:
        # The first frame is the frame of the test itself.
        continue

      # Both traceback_util_test and traceback_util are marked as excluded files
      # so we expect to see frames from both marked with __tracebackhide__.
      if filename == __file__ or filename == traceback_util.__file__:
        self.assertIn("__tracebackhide__", local_vars)
        self.assertTrue(local_vars["__tracebackhide__"])
    # Verify the frames are not actually removed.
    self.assertGreater(frame_count, 150)

  def test_wrapper_is_not_applied_twice(self):
    def f():
      pass

    f_wrapped = traceback_util.run_with_traceback_filter(f)
    f_wrapped_again = traceback_util.run_with_traceback_filter(f_wrapped)
    self.assertIs(f_wrapped, f_wrapped_again)

  def test_include_filename(self):
    self.assertFalse(
        traceback_util.include_filename("path/to/grain/_src/foo.py")
    )

    # Verify Windows path handling (simulated on non-Windows if needed)
    with mock.patch.object(pathlib, "Path", pathlib.PureWindowsPath):
      self.assertFalse(
          traceback_util.include_filename(r"C:\path\to\grain\_src\foo.py")
      )

    # Check that "grain" and "_src" must be adjacent
    self.assertTrue(
        traceback_util.include_filename("path/to/grain/something/_src/foo.py")
    )


class AddOneTransform(grain.transforms.Map):

  def map(self, x: int) -> int:
    return x + 1


class RaiseErrorTransform(grain.transforms.Map):

  def map(self, x: int) -> int:
    raise ValueError("Boom!")


class TracebackFilterTest(absltest.TestCase):

  def test_datasource_multiple_transforms_filters_traceback(self):
    config.update("py_traceback_filtering", "auto")
    range_ds = grain.sources.RangeDataSource(0, 10, 1)
    sampler = grain.samplers.IndexSampler(num_records=10, seed=42)
    ops = [RaiseErrorTransform()]
    for _ in range(100):
      ops.append(AddOneTransform())
    data_loader = grain.DataLoader(
        data_source=range_ds, sampler=sampler, operations=ops
    )
    _assert_exception_with_short_traceback(
        self, lambda: next(iter(data_loader)), ValueError
    )

  def test_dataset_multiple_transforms_filters_traceback(self):
    config.update("py_traceback_filtering", "auto")
    range_ds = grain.MapDataset.range(0, 10)
    range_ds = range_ds.map(RaiseErrorTransform())
    for _ in range(100):
      range_ds = range_ds.map(AddOneTransform())
    _assert_exception_with_short_traceback(
        self, lambda: next(iter(range_ds)), ValueError
    )


if __name__ == "__main__":
  absltest.main()
