"""Tests for the profiler.py when jax is linked."""

import os
import socket
from absl.testing import absltest
from grain._src.core import profiler


class ProfilerTest(absltest.TestCase):

  def test_framework(self):
    expected_framework = os.environ.get("EXPECTED_FRAMEWORK") or "jax"
    self.assertEqual(profiler.framework, expected_framework)

  def test_trace_annotation(self):
    if profiler.framework == profiler._NO_FRAMEWORK:
      self.assertIsNone(profiler.TraceAnnotation)
    else:
      self.assertIsNotNone(profiler.TraceAnnotation)
      with profiler.TraceAnnotation("test"):
        passes = True
      self.assertTrue(passes)

  def test_profiler_server(self):
    if profiler.framework == profiler._NO_FRAMEWORK:
      self.assertIsNone(profiler.start_server(1234))
    else:
      port = 1234
      profiler.start_server(port)
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", port))
      self.assertEqual(result, 0)
      profiler.stop_server()


if __name__ == "__main__":
  absltest.main()
