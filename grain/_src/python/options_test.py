import sys
from unittest import mock

from absl import logging
from grain._src.python import options

from absl.testing import absltest


class AutotuneOptionsTest(absltest.TestCase):

  def test_defaults(self):
    a = options.AUTOTUNE()
    self.assertEqual(a.min_value, 1)
    self.assertEqual(a.max_value, sys.float_info.max)
    self.assertEqual(a.initial_value, 1)

  def test_min_value_negative_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "min_value must be >= 0"):
      options.AUTOTUNE(min_value=-1)

  def test_max_value_less_than_min_value_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "max_value must be >= min_value"):
      options.AUTOTUNE(min_value=10, max_value=5)

  def test_initial_value_out_of_bounds_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, "initial_value must be >= min_value"
    ):
      options.AUTOTUNE(min_value=1, initial_value=0)
    with self.assertRaisesRegex(
        ValueError, "initial_value must be <= max_value"
    ):
      options.AUTOTUNE(max_value=10, initial_value=11)

  def test_valid_initialization(self):
    a = options.AUTOTUNE(min_value=1, max_value=10, initial_value=5)
    self.assertEqual(a.min_value, 1)
    self.assertEqual(a.max_value, 10)
    self.assertEqual(a.initial_value, 5)


class ReadOptionsTest(absltest.TestCase):

  def test_defaults(self):
    ro = options.ReadOptions()
    self.assertEqual(ro.num_threads, 16)
    self.assertEqual(ro.prefetch_buffer_size, 500)

  def test_num_threads_negative_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, "num_threads must be non-negative"):
      options.ReadOptions(num_threads=-1)

  def test_prefetch_buffer_size_negative_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, "prefetch_buffer_size must be non-negative"
    ):
      options.ReadOptions(prefetch_buffer_size=-1)

  def test_prefetch_buffer_size_less_than_num_threads_logs_warning(self):
    with self.assertLogs(level="WARNING") as logs:
      options.ReadOptions(num_threads=10, prefetch_buffer_size=5)
    self.assertIn(
        "prefetch_buffer_size=5 is smaller than num_threads=10", logs.output[0]
    )

  def test_prefetch_buffer_size_zero(self):
    with mock.patch.object(logging, "warning") as mock_warning:
      options.ReadOptions(num_threads=10, prefetch_buffer_size=0)
      mock_warning.assert_not_called()

  def test_autotune_values(self):
    ro = options.ReadOptions(
        num_threads=options.AUTOTUNE(min_value=1, max_value=10),
        prefetch_buffer_size=options.AUTOTUNE(
            min_value=10, max_value=100, initial_value=10
        ),
    )
    self.assertIsInstance(ro.num_threads, options.AUTOTUNE)
    self.assertIsInstance(ro.prefetch_buffer_size, options.AUTOTUNE)


if __name__ == "__main__":
  absltest.main()
