import logging
import re
from absl import logging as absl_logging
from grain._src.python import grain_logging
from absl.testing import absltest


class GrainLoggingTest(absltest.TestCase):

  def test_prefix_is_part_of_message(self):
    # self.assertLogs() doesn't format the messages, so we have to resort to
    # formatting directly with the absl handler to test whether the
    # prefix is being added.
    grain_logging.set_process_identifier_prefix('foo prefix')
    with self.assertLogs() as cm:
      absl_logging.info('example message')
    self.assertLen(cm.records, 1)
    log_record = cm.records[0]
    self.assertIn(
        'foo prefix', absl_logging.get_absl_handler().format(log_record)
    )

  def test_message_is_kept(self):
    grain_logging.set_process_identifier_prefix('Foo')
    with self.assertLogs() as cm:
      absl_logging.info('some info message: %i', 1337)
    self.assertLen(cm.output, 1)
    self.assertIn('some info message: 1337', cm.output[0])

  def test_message_formatting(self):
    log_record = logging.LogRecord(
        name=absl_logging.get_absl_logger().name,
        level=logging.INFO,
        pathname='file.cc',
        lineno=42,
        msg='some info message: %i',
        args=(789,),
        exc_info=None,
    )
    grain_logging.set_process_identifier_prefix('FooBarBaz 123')
    # We don't want to enforce a specific prefix format, but we want to avoid
    # duplicating the absl prefix (e.g.:
    # I0814 11:48:49.888083 1726756 grain_pool.py:161
    # ).
    self.assertTrue(
        re.search(
            r'.{0,5}FooBarBaz 123.{0,5} some info message: 789',
            absl_logging.get_absl_handler().format(log_record),
        )
    )


if __name__ == '__main__':
  absltest.main()
