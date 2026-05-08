"""Tests for the profiler.py when jax is linked."""

import os
import socket
import time
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import cloudpickle
import multiprocessing as mp
from grain._src.core import profiler
import portpicker


def _worker_main(worker_init_fn: bytes):
  """Helper function to start a profiler server in a subprocess."""
  worker_init_fn = cloudpickle.loads(worker_init_fn)
  worker_init_fn()
  time.sleep(10)


class ProfilerTest(absltest.TestCase):

  def test_framework(self):
    expected_framework = os.environ.get("EXPECTED_FRAMEWORK")
    self.assertEqual(profiler.get_framework(), expected_framework)

  def test_trace_annotation(self):
    self.assertIsNotNone(profiler.TraceAnnotation)
    with profiler.TraceAnnotation("test"):
      passes = True
    self.assertTrue(passes)

  def test_is_enabled(self):
    self.assertIsNotNone(profiler.is_enabled)
    self.assertFalse(profiler.is_enabled())

  def test_profiler_server(self):
    if not profiler.is_loaded():
      self.assertIsNone(profiler.start_server(1234))
    else:
      port = 1234
      profiler.start_server(port)
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", port))
      self.assertEqual(result, 0)
      profiler.stop_server()

  @flagsaver.flagsaver(grain_enable_multiprocess_worker_profiling=True)
  def test_register_unregister_subprocess(self):
    port = portpicker.pick_unused_port()
    mp_context = mp.get_context("spawn")
    subprocess = mp_context.Process(
        target=_worker_main,
        kwargs=dict({
            "worker_init_fn": cloudpickle.dumps(
                profiler.get_worker_init_fn(port)
            )
        }),
        daemon=True,
    )
    subprocess.start()
    if not profiler.is_worker_profiling_supported():
      with self.assertRaises(RuntimeError):
        profiler.register_subprocess(subprocess.pid, port)
    else:
      unregister_fn = profiler.register_subprocess(subprocess.pid, port)
      unregister_fn()
    subprocess.kill()

  @absltest.skipUnless(
      profiler.is_worker_profiling_supported(),
      "Worker profiling is not supported.",
  )
  @flagsaver.flagsaver(grain_enable_multiprocess_worker_profiling=True)
  def test_get_worker_init_fn_starts_server(self):
    port = portpicker.pick_unused_port()
    worker_init_fn = profiler.get_worker_init_fn(port)
    worker_init_fn()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      result = sock.connect_ex(("localhost", port))
    self.assertEqual(result, 0)
    profiler.stop_server()

  @flagsaver.flagsaver(grain_enable_multiprocess_worker_profiling=False)
  def test_get_worker_init_fn_does_not_start_server(self):
    port = portpicker.pick_unused_port()
    worker_init_fn = profiler.get_worker_init_fn(port)
    worker_init_fn()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      result = sock.connect_ex(("localhost", port))
    self.assertNotEqual(result, 0)

  @flagsaver.flagsaver
  def test_worker_profiling_enabled_flag(self):
    flags.FLAGS.grain_enable_multiprocess_worker_profiling = False
    self.assertFalse(profiler.is_worker_profiling_enabled())
    flags.FLAGS.grain_enable_multiprocess_worker_profiling = True
    self.assertEqual(
        profiler.is_worker_profiling_enabled(),
        profiler.is_worker_profiling_supported(),
    )


if __name__ == "__main__":
  absltest.main()
