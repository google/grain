# Copyright 2024 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
import multiprocessing as mp
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import interleave


_INTERLEAVE_TEST_CASES = (
    dict(
        testcase_name="cycle_length_1",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=1,
        expected=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    ),
    dict(
        testcase_name="cycle_length_2",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=2,
        expected=[1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 4, 5, 5, 5],
    ),
    dict(
        testcase_name="cycle_length_3",
        to_mix=[[1], [2, 2], [3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5, 5]],
        cycle_length=3,
        expected=[1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5, 5],
    ),
    dict(
        testcase_name="same_lengths",
        to_mix=[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        cycle_length=3,
        expected=[1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 4],
    ),
    dict(
        testcase_name="unsorted_lengths",
        to_mix=[[1, 1, 1], [2], [3, 3, 3, 3], [4, 4]],
        cycle_length=3,
        expected=[1, 2, 3, 1, 4, 3, 1, 4, 3, 3],
    ),
    dict(
        testcase_name="large_cycle_length",
        to_mix=[[1, 1, 1], [2], [3, 3, 3, 3], [4, 4]],
        cycle_length=10,
        expected=[1, 2, 3, 4, 1, 3, 4, 1, 3, 3],
    ),
)


class MixedIterDatasetTest(parameterized.TestCase):

  @parameterized.named_parameters(*_INTERLEAVE_TEST_CASES)
  def test_interleaved_mix(self, to_mix, cycle_length, expected):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in to_mix
    ]
    ds = interleave.InterleaveIterDataset(datasets, cycle_length=cycle_length)
    self.assertEqual(list(ds), expected)
    # Sanity check.
    flat_inputs = []
    for ds in datasets:
      flat_inputs.extend(list(ds))
    self.assertCountEqual(flat_inputs, expected)

  @parameterized.named_parameters(*_INTERLEAVE_TEST_CASES)
  def test_checkpoint(self, to_mix, cycle_length, expected):
    datasets = [
        dataset.MapDataset.source(elements).to_iter_dataset()
        for elements in to_mix
    ]
    ds = interleave.InterleaveIterDataset(datasets, cycle_length=cycle_length)
    ds_iter = ds.__iter__()
    checkpoints = {}
    for i in range(len(expected)):
      checkpoints[i] = ds_iter.get_state()
      _ = next(ds_iter)
    for i, state in checkpoints.items():
      ds_iter.set_state(state)
      self.assertEqual(
          list(ds_iter), expected[i:], msg=f"Failed at checkpoint {i}."
      )

  def test_with_map_dataset_of_datasets(self):

    def make_dummy_source(filename):
      chars = [c for c in filename]
      return dataset.MapDataset.source(chars)

    filenames = dataset.MapDataset.source(["11", "2345", "678", "9999"])
    sources = filenames.shuffle(seed=42).map(make_dummy_source)
    ds = interleave.InterleaveIterDataset(sources, cycle_length=2)
    self.assertEqual(
        list(ds),
        ["1", "2", "1", "3", "6", "4", "7", "5", "8", "9", "9", "9", "9"],
    )

  def test_with_mp_prefetch(self):
    ds = dataset.MapDataset.range(1, 6).map(
        lambda i: dataset.MapDataset.source([i]).repeat(i).to_iter_dataset()
    )
    ds = interleave.InterleaveIterDataset(ds, cycle_length=5)
    ds = ds.mp_prefetch(options.MultiprocessingOptions(num_workers=3))
    self.assertEqual(list(ds), [1, 2, 3, 4, 5, 3, 4, 2, 3, 4, 5, 4, 5, 5, 5])


if __name__ == "__main__":
  absltest.main()
