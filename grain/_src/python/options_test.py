import platform
from unittest import mock

from absl import logging
import multiprocessing as mp
from grain._src.python import data_sources
from grain._src.python import options
from grain._src.python.dataset import dataset
import numpy as np

from absl.testing import absltest


class ReadOptionsTest(absltest.TestCase):

  def test_defaults(self):
    ro = options.ReadOptions()
    self.assertEqual(ro.num_threads, 16)
    self.assertEqual(ro.prefetch_buffer_size, 500)

  def test_example_docstring(self):
    source = data_sources.RangeDataSource(0, 4, 1)
    read_options = options.ReadOptions(
        num_threads=8,
        prefetch_buffer_size=100,
    )
    ds = (
        dataset.MapDataset.source(source)
        .to_iter_dataset(read_options=read_options)
        .batch(2)
    )
    np.testing.assert_equal(
        list(ds),
        [
            np.array([0, 1]),
            np.array([2, 3]),
        ],
    )

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


@absltest.skipIf(platform.system() == "Windows", "Timeouts on Windows.")
class MultiprocessingOptionsTest(absltest.TestCase):

  def test_example_docstring(self):
    read_options = options.ReadOptions(
        num_threads=8,
        prefetch_buffer_size=10,
    )
    multiprocessing_options = options.MultiprocessingOptions(
        num_workers=2,
        per_worker_buffer_size=10,
    )
    ds = (
        dataset.MapDataset.range(4)
        .to_iter_dataset(read_options=read_options)
        .batch(2)
        .mp_prefetch(multiprocessing_options)
    )
    np.testing.assert_equal(
        list(ds),
        [
            np.array([0, 2]),
            np.array([1, 3]),
        ],
    )


if __name__ == "__main__":
  absltest.main()
