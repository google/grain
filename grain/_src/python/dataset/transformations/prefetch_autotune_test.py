import sys
from unittest import mock

from absl.testing import absltest
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch_autotune
import numpy as np

from absl.testing import absltest

PerformanceConfig = prefetch_autotune.PerformanceConfig


class PrefetchAutotuneTest(absltest.TestCase):

  def test_get_element_size_numpy(self):
    element = np.zeros((10, 20), dtype=np.int32)
    self.assertEqual(
        prefetch_autotune._get_element_size_bytes(element), element.nbytes
    )

  def test_get_element_size_dict(self):
    element = {
        'a': np.ones(5, dtype=np.float64),
        'b': np.zeros(3, dtype=np.int16),
    }
    expected_size = element['a'].nbytes + element['b'].nbytes
    self.assertEqual(
        prefetch_autotune._get_element_size_bytes(element), expected_size
    )

  def test_get_element_size_list(self):
    element = [np.ones(4), 'test_string']
    expected_size = element[0].nbytes + sys.getsizeof(element[1])
    self.assertEqual(
        prefetch_autotune._get_element_size_bytes(element), expected_size
    )

  @mock.patch.object(prefetch_autotune, 'cpu_count', return_value=16)
  def test_autotune_calculates_workers_correctly(self, mock_cpu_count):
    # 100 MB per element. With a 1024 MB budget, we expect 10 workers.
    element = np.zeros(100 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 10).to_iter_dataset()

    performance_config = prefetch_autotune.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024,
        max_workers=None,  # Should default to cpu_count
    )
    mock_cpu_count.assert_called_once()
    self.assertEqual(performance_config.multiprocessing_options.num_workers, 10)

  def test_autotune_respects_max_workers_cap(self):
    # 10 MB per element. Budget allows for 102 workers, but cap is 8.
    element = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 10).to_iter_dataset()

    performance_config = prefetch_autotune.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024,
        max_workers=8,
    )
    # The calculated value (102) is capped at max_workers (8).
    self.assertEqual(performance_config.multiprocessing_options.num_workers, 8)

  @mock.patch.object(prefetch_autotune, 'cpu_count', return_value=12)
  def test_autotune_with_insufficient_samples_defaults_to_max_workers(
      self, mock_cpu_count
  ):
    # Dataset has fewer elements than samples_to_check (default 5).
    ds = dataset.MapDataset.source([1, 2, 3]).to_iter_dataset()

    performance_config = prefetch_autotune.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024,
        max_workers=None,
    )
    # Should default to cpu_count when sampling fails.
    mock_cpu_count.assert_called_once()
    self.assertEqual(performance_config.multiprocessing_options.num_workers, 12)

  def test_get_max_workers_calculates_from_mean_element_size(self):
    # Create elements of different sizes (100MB, 200MB, 300MB) to test the
    # averaging logic. The expected average size is 200MB.
    elements = [
        np.zeros(100 * 1024 * 1024, dtype=np.uint8),
        np.zeros(200 * 1024 * 1024, dtype=np.uint8),
        np.zeros(300 * 1024 * 1024, dtype=np.uint8),
    ]
    ds = dataset.MapDataset.source(elements).to_iter_dataset()

    # With a RAM budget of 1024 MB and an avg element size of 200MB, we
    # expect to fit 1024 / 100 ~ 10 workers.
    expected_workers = 10

    # Mock cpu_count to be higher than the expected result to ensure our
    # calculation is the limiting factor.
    with mock.patch.object(prefetch_autotune, 'cpu_count', return_value=32):
      num_workers = prefetch_autotune._get_max_workers(
          ds,
          ram_budget_mb=1024,
          max_workers=None,
          samples_to_check=1,
      )
    self.assertEqual(num_workers, expected_workers)

  def test_autotune_passes_worker_init_fn(self):
    ds = dataset.MapDataset.source([1, 2, 3]).to_iter_dataset()

    # Mock _get_max_workers to avoid actual calculation.
    with mock.patch.object(
        prefetch_autotune, '_get_max_workers', return_value=4
    ) as mock_get_max:
      performance_config = prefetch_autotune.pick_performance_config(
          ds=ds,
          ram_budget_mb=512,
          max_workers=8,
      )
      mock_get_max.assert_called_once()

    self.assertEqual(performance_config.multiprocessing_options.num_workers, 4)


if __name__ == '__main__':
  absltest.main()
