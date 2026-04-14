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
from absl.testing import parameterized
import cloudpickle
import grain
from grain._src.core import traceback_util
from grain._src.core.config import config  # pylint: disable=g-importing-member


def _assert_exception_with_short_traceback(
    self: absltest.TestCase,
    fn: Callable[[], None],
    expected_error_type: type[BaseException],
    final_raise_function_name: str,
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
    final_raise_function_name: The name of the function that is expected to
      raise the exception.
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
  # Check that we unconditionally include the frame that raised the exception.
  self.assertEqual(tb[-1].name, final_raise_function_name)


def raise_an_error():
  raise ValueError("Boom!")


def raise_after_recursion(curr_count: int, max_count: int) -> None:
  if curr_count == max_count:
    raise_an_error()
  raise_after_recursion(curr_count + 1, max_count)


# Set decorator outside of the recursive function so it is only applied once.
@traceback_util.run_with_traceback_filter
def start_raise_after_recursion(curr_count: int, max_count: int) -> None:
  raise_after_recursion(curr_count, max_count)


# Set decorator on the recursive function so it is applied on multiple calls.
@traceback_util.run_with_traceback_filter
def raise_after_recursion_with_traceback_filter(
    curr_count: int, max_count: int
) -> None:
  if curr_count == max_count:
    raise_an_error()
  raise_after_recursion(curr_count + 1, max_count)


class TracebackUtilTest(parameterized.TestCase):

  @parameterized.parameters(
      ("auto", start_raise_after_recursion),
      ("auto", raise_after_recursion_with_traceback_filter),
      ("remove_frames", start_raise_after_recursion),
      ("remove_frames", raise_after_recursion_with_traceback_filter),
      ("quiet_remove_frames", start_raise_after_recursion),
      ("quiet_remove_frames", raise_after_recursion_with_traceback_filter),
  )
  def test_traceback_mode_decorator_filters_traceback(
      self, traceback_filtering_mode: str, fn: Callable[[int, int], None]
  ):
    config.update("py_traceback_filtering", traceback_filtering_mode)
    _assert_exception_with_short_traceback(
        self,
        lambda: fn(0, 150),
        ValueError,
        "raise_an_error",
    )

  @parameterized.parameters(
      start_raise_after_recursion,
      raise_after_recursion_with_traceback_filter,
  )
  def test_traceback_mode_off_decorator_does_not_filter_traceback(
      self, fn: Callable[[int, int], None]
  ):
    config.update("py_traceback_filtering", "off")
    try:
      fn(0, 150)
      self.fail("Expected ValueError to be raised.")
    except ValueError as e:
      tb = traceback.extract_tb(e.__traceback__)
    self.assertGreater(len(tb), 150)

  @parameterized.parameters(
      start_raise_after_recursion,
      raise_after_recursion_with_traceback_filter,
  )
  def test_traceback_mode_tracebackhide_decorator_applies_tracebackhide(
      self, fn: Callable[[int, int], None]
  ):
    config.update("py_traceback_filtering", "tracebackhide")
    frame_locals = []
    try:
      fn(0, 150)
      self.fail("Expected ValueError to be raised.")
    except ValueError as e:
      frames = list(traceback.walk_tb(e.__traceback__))
      for f, _ in reversed(frames):
        frame_locals.append(
            (f.f_code.co_filename, f.f_code.co_name, f.f_locals)
        )
    frame_count = 0
    # Verify expected frames are marked with __tracebackhide__.
    for filename, funcname, local_vars in frame_locals[:-1]:
      frame_count += 1

      # Both traceback_util_test and traceback_util are marked as excluded files
      # so we expect to see frames from both marked with __tracebackhide__.
      # The most recent frame that raised the exception is never filtered.
      if (
          filename == __file__ or filename == traceback_util.__file__
      ) and funcname != "raise_an_error":
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

  def test_reconstruct_traceback(self):
    def inner_func():
      unused_x = 42  # pylint: disable=unused-variable
      raise ValueError("original error")

    # We cannot use self.assertRaises because it clears the traceback.
    try:
      inner_func()
      self.fail("Expected ValueError to be raised.")
    except ValueError as e:
      original_tb_obj = traceback_util.PicklableTraceback.from_traceback(
          e.__traceback__
      )

    # Test serialization
    pickled = cloudpickle.dumps(original_tb_obj)
    unpickled_tb_obj = cloudpickle.loads(pickled)
    self.assertEqual(original_tb_obj, unpickled_tb_obj)

    # Test reconstruction
    reconstructed_tb = traceback_util.reconstruct_traceback(unpickled_tb_obj)
    self.assertIsNotNone(reconstructed_tb)

    # Verify reconstructed frames
    reconstructed_summary = list(traceback.walk_tb(reconstructed_tb))
    self.assertEqual(len(reconstructed_summary), len(original_tb_obj.frames))

    for i, (f, lineno) in enumerate(reconstructed_summary):
      self.assertEqual(f.f_code.co_filename, original_tb_obj.frames[i].filename)
      expected_name = original_tb_obj.frames[i].name
      if expected_name == "<module>":
        expected_name = "module"
      elif not expected_name.isidentifier():
        expected_name = "dummy_func"

      self.assertEqual(f.f_code.co_name, expected_name)
      self.assertEqual(lineno, original_tb_obj.frames[i].lineno)

  def test_from_traceback_none(self):
    tb_obj = traceback_util.PicklableTraceback.from_traceback(None)
    self.assertEmpty(tb_obj.frames)

  def test_reconstruct_traceback_compilation_failure_is_skipped(self):
    # 'class' is a valid identifier but a reserved keyword.
    # Compiling `def class(): ...` will raise a SyntaxError.
    # We expect this frame to be logged as a warning and skipped, while
    # the valid frames around it should still be reconstructed successfully.
    invalid_frame = traceback_util.PicklableFrame(
        filename="invalid.py", lineno=10, name="class", line=None
    )
    valid_frame_1 = traceback_util.PicklableFrame(
        filename="valid1.py", lineno=5, name="good_func_1", line=None
    )
    valid_frame_2 = traceback_util.PicklableFrame(
        filename="valid2.py", lineno=15, name="good_func_2", line=None
    )

    worker_tb = traceback_util.PicklableTraceback(
        frames=[valid_frame_1, invalid_frame, valid_frame_2]
    )

    reconstructed_tb = traceback_util.reconstruct_traceback(worker_tb)
    self.assertIsNotNone(reconstructed_tb)

    # The reconstructed traceback should only contain the 2 valid frames.
    reconstructed_summary = list(traceback.walk_tb(reconstructed_tb))
    self.assertLen(reconstructed_summary, 2)
    self.assertEqual(reconstructed_summary[0][0].f_code.co_name, "good_func_1")
    self.assertEqual(reconstructed_summary[1][0].f_code.co_name, "good_func_2")

  def test_reconstruct_traceback_special_names(self):
    special_names = [
        ("<module>", "module"),
        ("<listcomp>", "listcomp"),
        ("<dictcomp>", "dictcomp"),
        ("<setcomp>", "setcomp"),
        ("<genexpr>", "genexpr"),
        ("<lambda>", "lambda_func"),
        ("invalid name!", "dummy_func"),
    ]
    frames = [
        traceback_util.PicklableFrame(
            filename=f"file_{i}.py", lineno=i + 1, name=original_name, line=None
        )
        for i, (original_name, _) in enumerate(special_names)
    ]
    worker_tb = traceback_util.PicklableTraceback(frames=frames)

    reconstructed_tb = traceback_util.reconstruct_traceback(worker_tb)
    self.assertIsNotNone(reconstructed_tb)

    reconstructed_summary = list(traceback.walk_tb(reconstructed_tb))
    self.assertLen(reconstructed_summary, len(special_names))

    for i, (_, expected_name) in enumerate(special_names):
      self.assertEqual(
          reconstructed_summary[i][0].f_code.co_name, expected_name
      )


class AddOneTransform(grain.transforms.Map):

  def map(self, x: int) -> int:
    return x + 1


class RaiseErrorTransform(grain.transforms.Map):

  def raise_error_in_transform(self):
    raise ValueError("Boom!")

  def map(self, x: int) -> int:
    self.raise_error_in_transform()


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
        self,
        lambda: next(iter(data_loader)),
        ValueError,
        "raise_error_in_transform",
    )

  def test_dataset_multiple_transforms_filters_traceback(self):
    config.update("py_traceback_filtering", "auto")
    range_ds = grain.MapDataset.range(0, 10)
    range_ds = range_ds.map(RaiseErrorTransform())
    for _ in range(100):
      range_ds = range_ds.map(AddOneTransform())
    _assert_exception_with_short_traceback(
        self,
        lambda: next(iter(range_ds)),
        ValueError,
        "raise_error_in_transform",
    )


if __name__ == "__main__":
  absltest.main()
