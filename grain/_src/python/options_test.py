from unittest import mock

from absl import logging
from grain._src.python import options

from absl.testing import absltest


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


if __name__ == "__main__":
  absltest.main()
