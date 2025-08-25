import sys
from unittest import mock

from absl.testing import absltest
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch_autotune
from grain._src.python.dataset.transformations.map import MapMapDataset
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
        max_buffer_size=None,
    )
    mock_cpu_count.assert_called_once()
    self.assertEqual(performance_config.multiprocessing_options.num_workers, 10)
    self.assertEqual(performance_config.read_options.prefetch_buffer_size, 10)
    self.assertEqual(performance_config.read_options.num_threads, 16)

  def test_autotune_respects_max_workers_cap(self):
    # 10 MB per element. Budget allows for 102 workers, but cap is 8.
    element = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 10).to_iter_dataset()

    performance_config = prefetch_autotune.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024,
        max_workers=8,
        max_buffer_size=None,
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
        max_buffer_size=None,
    )
    # Should default to cpu_count when sampling fails.
    mock_cpu_count.assert_called_once()
    self.assertEqual(performance_config.multiprocessing_options.num_workers, 12)

  def test_get_num_workers_calculates_from_mean_element_size(self):
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
      num_workers = prefetch_autotune._get_num_workers(
          ds,
          ram_budget_mb=1024,
          max_workers=None,
          samples_to_check=1,
      )
    self.assertEqual(num_workers, expected_workers)

  def test_autotune_passes_worker_init_fn(self):
    ds = dataset.MapDataset.source([1, 2, 3]).to_iter_dataset()

    # Mock _get_num_workers to avoid actual calculation.
    # Mock _get_num_workers to avoid actual calculation.
    with mock.patch.object(
        prefetch_autotune, '_get_num_workers', return_value=4
    ) as mock_get_max:
      performance_config = prefetch_autotune.pick_performance_config(
          ds=ds,
          ram_budget_mb=512,
          max_workers=8,
          max_buffer_size=None,
      )
      mock_get_max.assert_called_once()

    self.assertEqual(performance_config.multiprocessing_options.num_workers, 4)

  def test_get_buffer_size_calculates_correctly(self):
    # Each element is 10MB
    element = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 10).to_iter_dataset()

    # With a 100MB budget and 10MB elements, we expect buffer size of 10.
    read_options = prefetch_autotune._get_buffer_size(
        ds,
        ram_budget_mb=100,
        max_buffer_size=10,
        samples_to_check=5,
    )
    self.assertEqual(read_options.num_threads, 16)
    self.assertEqual(read_options.prefetch_buffer_size, 10)

  def test_get_buffer_size_respects_max_threads_cap(self):
    element = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 10).to_iter_dataset()

    # Calculation would yield buffer size of 10, but it should be capped at 3.
    read_options = prefetch_autotune._get_buffer_size(
        ds,
        ram_budget_mb=100,
        max_buffer_size=3,
        samples_to_check=5,
    )
    self.assertEqual(read_options.num_threads, 16)
    self.assertEqual(read_options.prefetch_buffer_size, 3)

  def test_get_buffer_size(self):
    element = np.zeros(10 * 1024 * 1024, dtype=np.uint8)
    ds = dataset.MapDataset.source([element] * 5).to_iter_dataset()

    read_options = prefetch_autotune._get_buffer_size(
        ds,
        ram_budget_mb=500,
        max_buffer_size=None,
        samples_to_check=3,
    )
    self.assertEqual(read_options.num_threads, 16)
    self.assertEqual(read_options.prefetch_buffer_size, 50)

  def test_find_prefetch_iter_dataset_parent_returns_parent(self):
    ds = (
        dataset.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .map(lambda x: x + 1)
        .to_iter_dataset()
        .batch(2)
    )
    parent_prefetch_ds = prefetch_autotune._find_prefetch_iter_dataset_parents(
        ds
    )
    # Verify the type as the instance accounts for inheritance.
    self.assertLen(parent_prefetch_ds, 1)
    self.assertEqual(type(parent_prefetch_ds[0]), MapMapDataset)

  def test_find_prefetch_iter_dataset_parent_no_prefetch_iter_dataset(self):
    ds = (
        dataset.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .map(lambda x: x + 1)
        .batch(2)
    )
    parent_prefetch_ds = prefetch_autotune._find_prefetch_iter_dataset_parents(
        ds
    )
    self.assertEmpty(parent_prefetch_ds)

  def test_find_prefetch_iter_dataset_parent_multiple_parents(self):
    ds1 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .map(lambda x: x + 1)
        .to_iter_dataset()
        .batch(2)
    )
    ds2 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5])
        .map(lambda x: x + 2)
        .to_iter_dataset()
        .batch(5)
    )
    ds3 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5])
        .map(lambda x: x + 3)
        .to_iter_dataset()
        .batch(2)
    )
    ds4 = dataset.IterDataset.mix([ds1, ds2])
    mixed_dataset = dataset.IterDataset.mix([ds3, ds4])
    parents_prefetch_ds = prefetch_autotune._find_prefetch_iter_dataset_parents(
        mixed_dataset
    )
    self.assertLen(parents_prefetch_ds, 3)
    self.assertEqual(list(parents_prefetch_ds[0]), [4, 5, 6, 7, 8])
    self.assertEqual(
        list(parents_prefetch_ds[1]), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    self.assertEqual(list(parents_prefetch_ds[2]), [3, 4, 5, 6, 7])

  def test_find_prefetch_iter_dataset_parent_multiple_parents_with_empty(self):
    ds1 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .map(lambda x: x + 1)
        .to_iter_dataset()
        .batch(2)
    )
    ds2 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5]).map(lambda x: x + 2).batch(5)
    )
    ds3 = (
        dataset.MapDataset.source([1, 2, 3, 4, 5])
        .map(lambda x: x + 3)
        .to_iter_dataset()
        .batch(2)
    )
    ds4 = dataset.IterDataset.mix([ds1, ds2])
    mixed_dataset = dataset.IterDataset.mix([ds3, ds4])
    parents_prefetch_ds = prefetch_autotune._find_prefetch_iter_dataset_parents(
        mixed_dataset
    )
    self.assertLen(parents_prefetch_ds, 2)
    self.assertEqual(list(parents_prefetch_ds[0]), [4, 5, 6, 7, 8])
    self.assertEqual(
        list(parents_prefetch_ds[1]), [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )

  def test_get_average_element_size_mb_returns_zero_for_empty_dataset(self):
    self.assertEqual(prefetch_autotune._get_average_element_size_mb([]), 0)

  def test_get_average_element_size_mb(self):
    element1 = np.zeros(1 * 1024 * 1024, dtype=np.uint8)  # 1MB
    element2 = np.zeros(3 * 1024 * 1024, dtype=np.uint8)  # 3MB
    element3 = np.zeros(5 * 1024 * 1024, dtype=np.uint8)  # 5MB
    ds1 = dataset.MapDataset.source(
        [element1, element2, element3]
    ).to_iter_dataset()
    ds2 = dataset.MapDataset.source([element1] * 10).to_iter_dataset()
    ds3 = dataset.MapDataset.source([element2, element3] * 5).to_iter_dataset()

    # samples_to_check = 5. samples_per_ds = ceil(5/3) = 2.
    # ds1: 2 samples of size (1 and 3). Total = 4MB.
    # ds2: 2 samples of size 1MB. Total = 2MB.
    # ds3: 2 samples of size 3 and 5. Total = 8MB.
    # Expected average = (4 + 2 + 8) / 6 = 2.333...
    avg_size_mb = prefetch_autotune._get_average_element_size_mb(
        [ds1, ds2, ds3], samples_to_check=6
    )
    self.assertAlmostEqual(avg_size_mb, 14 / 6)

  def test_get_average_element_size_mixed_ds_mb(self):
    element1 = np.zeros(1 * 1024 * 1024, dtype=np.uint8)  # 1MB
    element2 = np.zeros(3 * 1024 * 1024, dtype=np.uint8)  # 3MB
    element3 = np.zeros(5 * 1024 * 1024, dtype=np.uint8)  # 5MB
    ds1 = dataset.MapDataset.source(
        [element1, element2, element3]
    ).to_iter_dataset()
    ds2 = dataset.MapDataset.source([element1] * 10).to_iter_dataset()
    ds3 = dataset.MapDataset.source([element2, element3] * 5).to_iter_dataset()
    ds4 = dataset.IterDataset.mix([ds1, ds2])
    mixed_ds = dataset.IterDataset.mix([ds3, ds4])

    # samples_to_check = 5
    # There are 3 root datasets with non-batched elements, consisting of:
    # ds3 = [3, 5, 3, 5, 3, 5, 3, 5, 3, 5], d1 = [1, 3, 5], ds2 = [1, ... 1]
    # The average size is (3 + 5 + 1 + 3 + 1 + 1) / 6 = 2.3364
    performance_config = prefetch_autotune.pick_performance_config(
        mixed_ds, ram_budget_mb=500, max_workers=None, max_buffer_size=None
    )
    self.assertEqual(performance_config.read_options.prefetch_buffer_size, 71)

if __name__ == '__main__':
  absltest.main()
