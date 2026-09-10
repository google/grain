# Copyright 2023 Google LLC
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
"""Tests for shuffle transformation."""

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import shuffle
import numpy as np


class ShuffleMapDatasetTest(parameterized.TestCase):

  def test_len(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    self.assertLen(ds, 400)

  def test_getitem(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertCountEqual(shuffled_indices, list(range(400)))

    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertCountEqual(shuffled_indices_epoch2, list(range(400)))
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_getitems(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    shuffled_indices = ds._getitems(range(400))
    self.assertCountEqual(shuffled_indices, list(range(400)))
    self.assertEqual(shuffled_indices, [ds[i] for i in range(400)])

    shuffled_indices_epoch2 = ds._getitems(range(400, 800))
    self.assertCountEqual(shuffled_indices_epoch2, list(range(400)))
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)
    self.assertEqual(shuffled_indices_epoch2, [ds[400 + i] for i in range(400)])

  def test_iter(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    self.assertLen(elements, 400)

  def test_default_seed(self):
    seed = 42
    ds1 = dataset.MapDataset.range(400).seed(seed)
    ds1 = shuffle.ShuffleMapDataset(ds1)
    ds2 = dataset.MapDataset.range(400).seed(seed)
    ds2 = shuffle.ShuffleMapDataset(ds2)
    self.assertEqual(list(ds1), list(ds2))

  def test_raises_with_no_seed(self):
    with self.assertRaises(ValueError):
      shuffle.ShuffleMapDataset(dataset.MapDataset.range(400))

  @parameterized.parameters(-1000, -1, 2**32, 2**32 + 1, 2**64 + 1)
  def test_init_with_invalid_seed_returns_value_error(self, seed):
    with self.assertRaises(ValueError):
      shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=seed)

  def test_element_spec(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(2), seed=42)
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())

  def test_shuffled_index_with_numpy_integer_does_not_overflow(self):
    ds = shuffle.ShuffleMapDataset(dataset.MapDataset.range(400), seed=42)
    # Prevents regression of xid/273443246
    for dtype in [np.int32, np.uint32, np.int64]:
      index = dtype(5)
      # Should not raise OverflowError when computing modulo 2**32
      _ = ds._shuffled_index(index)  # pytype: disable=wrong-arg-types

  def test_cross_version_determinism(self):
    # This test validates shuffle determinism across different versions of
    # Grain given a fixed seed. Note that we technically do not guarantee
    # cross-version determinism, but multiple users nevertheless rely on it
    # because it holds in practice. Any updates to the shuffle code could break
    # it. Only update the values if you know what you're doing.
    ds = dataset.MapDataset.range(10).seed(42)
    ds = shuffle.ShuffleMapDataset(ds)
    self.assertEqual(list(ds), [1, 7, 6, 9, 0, 8, 4, 5, 3, 2])


class WindowShuffleMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400), window_size=10, seed=42
    )
    self.assertLen(ds, 400)

  def test_getitem(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400),
        window_size=window_size,
        seed=42,
    )
    shuffled_indices = [ds[i] for i in range(400)]
    self.assertLen(shuffled_indices, 400)
    for i in range(0, 400, 10):
      self.assertBetween(shuffled_indices[i], i, i + (window_size - 1))

  def test_getitem_multi_epochs(self):
    # Multiple epochs shouldn't affect window shuffling.
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400),
        window_size=10,
        seed=42,
    )
    shuffled_indices = [ds[i] for i in range(400)]
    shuffled_indices_epoch2 = [ds[400 + i] for i in range(400)]
    self.assertLen(shuffled_indices, 400)
    self.assertLen(shuffled_indices_epoch2, 400)
    self.assertNotEqual(shuffled_indices, shuffled_indices_epoch2)

  def test_getitems(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400), window_size=window_size, seed=42
    )
    shuffled_indices = ds._getitems(range(400))
    self.assertLen(shuffled_indices, 400)
    for i in range(0, 400, window_size):
      # Check that elements in each window are shuffled but stay within their
      # original window bounds.
      window_elements = shuffled_indices[i : i + window_size]
      original_window_elements = list(range(i, i + window_size))
      self.assertCountEqual(window_elements, original_window_elements)
    # Check consistency with __getitem__
    self.assertEqual(shuffled_indices, [ds[i] for i in range(400)])

  def test_getitems_multi_epochs(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400),
        window_size=window_size,
        seed=42,
    )
    shuffled_indices_epoch1 = ds._getitems(range(400))
    shuffled_indices_epoch2 = ds._getitems(range(400, 800))
    self.assertLen(shuffled_indices_epoch1, 400)
    self.assertLen(shuffled_indices_epoch2, 400)
    self.assertNotEqual(shuffled_indices_epoch1, shuffled_indices_epoch2)

    for i in range(0, 400, window_size):
      # Check that elements in each window are shuffled but stay within their
      # original window bounds for epoch 1.
      window_elements_epoch1 = shuffled_indices_epoch1[i : i + window_size]
      original_window_elements = list(range(i, i + window_size))
      self.assertCountEqual(window_elements_epoch1, original_window_elements)

      # Check that elements in each window are shuffled but stay within their
      # original window bounds for epoch 2.
      window_elements_epoch2 = shuffled_indices_epoch2[i : i + window_size]
      self.assertCountEqual(window_elements_epoch2, original_window_elements)

  def test_iter(self):
    window_size = 10
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400), window_size=window_size, seed=42
    )
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(400)]
    for i in range(0, 400, 10):
      self.assertBetween(elements[i], i, i + (window_size - 1))

  def test_example_docstring(self):
    window_size = 3
    parent_ds = dataset.MapDataset.range(12)

    shuffled_ds = shuffle.WindowShuffleMapDataset(
        parent_ds,
        window_size=window_size,
        seed=42,
    )
    parent = list(parent_ds)
    shuffled = list(shuffled_ds)

    # All elements are preserved.
    self.assertCountEqual(shuffled, parent)
    # Elements remain within their original window.
    for start in range(0, len(parent), window_size):
      self.assertCountEqual(
          shuffled[start : start + window_size],
          parent[start : start + window_size],
      )

  def test_element_spec(self):
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(20), window_size=5, seed=42
    )
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())

  def test_shuffled_index_with_numpy_integer_does_not_overflow(self):
    ds = shuffle.WindowShuffleMapDataset(
        dataset.MapDataset.range(400), window_size=10, seed=42
    )
    # Prevents regression of xid/273443246
    for dtype in [np.int32, np.uint32, np.int64]:
      index = dtype(5)
      _ = ds._shuffled_index(index)  # pytype: disable=wrong-arg-types


class WindowShuffleInterDatasetTest(parameterized.TestCase):
  _DATASET_SIZE = 30
  _WINDOW_SIZE = 10

  def setUp(self):
    super().setUp()
    self.range_iter_ds = dataset.MapDataset.range(
        0, self._DATASET_SIZE
    ).to_iter_dataset()

  def test_elements_are_shuffled(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=self._WINDOW_SIZE, seed=42
    )
    original_elements = list(self.range_iter_ds)
    shuffled_elements = list(ds)
    self.assertNotEqual(original_elements, shuffled_elements)
    self.assertCountEqual(original_elements, shuffled_elements)

  def test_example_docstring(self):
    window_size = 3
    parent_ds = dataset.MapDataset.range(12).to_iter_dataset()
    parent = list(parent_ds)
    shuffled_ds = shuffle.WindowShuffleIterDataset(
        parent_ds,
        window_size=window_size,
        seed=42,
    )
    shuffled = list(shuffled_ds)
    # All elements are preserved.
    self.assertCountEqual(shuffled, parent)
    # Elements remain within their original window.
    for start in range(0, len(parent), window_size):
      self.assertCountEqual(
          shuffled[start : start + window_size],
          parent[start : start + window_size],
      )

  def test_different_seeds_used_between_windows(self):
    window_size = 8
    original_ds = dataset.MapDataset.range(window_size).repeat(3)
    shuffled_ds = shuffle.WindowShuffleIterDataset(
        original_ds.to_iter_dataset(), window_size=window_size, seed=42
    )
    shuffled_ds_iter = shuffled_ds.__iter__()
    first_window = [next(shuffled_ds_iter) for _ in range(window_size)]
    second_window = [next(shuffled_ds_iter) for _ in range(window_size)]
    third_window = [next(shuffled_ds_iter) for _ in range(window_size)]
    self.assertNotEqual(first_window, second_window)
    self.assertNotEqual(second_window, third_window)
    self.assertNotEqual(first_window, third_window)
    self.assertCountEqual(first_window, second_window)
    self.assertCountEqual(second_window, third_window)
    self.assertCountEqual(first_window, third_window)

  def test_shuffled_checkpoints(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=4, seed=42
    )
    ds_iter = ds.__iter__()
    checkpoints = []
    checkpoints_values = []
    for _ in range(self._DATASET_SIZE):
      checkpoints.append(ds_iter.get_state())
      checkpoints_values.append(next(ds_iter))
    for i in range(len(checkpoints)):
      ds_iter.set_state(checkpoints[i])
      self.assertDictEqual(
          ds_iter.get_state(),
          checkpoints[i],
          msg=f"Checkpoint state failed at {i}.",
      )
      values = list(ds_iter)
      self.assertEqual(
          checkpoints_values[i:],
          values,
          msg=f"Checkpoint values failed from checkpoint {i}.",
      )

  def test_checkpoint_restore_on_fresh_iterator(self):
    window_size = 10
    num_elements = 1000
    seed = 42
    num_full_windows_before_checkpoint = 5
    num_elements_to_verify = 20

    # Checkpoint position: halfway through a window to trigger the bug
    # (need to be partway through a window so next() triggers window refill)
    checkpoint_position = (
        num_full_windows_before_checkpoint * window_size + window_size // 2
    )

    # Create dataset with enough elements to span multiple windows
    ds = dataset.MapDataset.range(num_elements).to_iter_dataset()
    ds = shuffle.WindowShuffleIterDataset(
        ds, window_size=window_size, seed=seed
    )

    # Original continuous run: consume to checkpoint position
    it1 = ds.__iter__()
    for _ in range(checkpoint_position):
      next(it1)
    checkpoint_state = it1.get_state()

    # Continue and record data for verification
    elements_after_checkpoint_original = [
        next(it1) for _ in range(num_elements_to_verify)
    ]
    state_after_verification_original = it1.get_state()

    # Now simulate checkpoint restore from a fresh iterator
    ds2 = dataset.MapDataset.range(num_elements).to_iter_dataset()
    ds2 = shuffle.WindowShuffleIterDataset(
        ds2, window_size=window_size, seed=seed
    )
    it2 = ds2.__iter__()

    # Restore state at checkpoint position
    it2.set_state(checkpoint_state)

    # Continue from checkpoint and verify data matches
    elements_after_checkpoint_restored = [
        next(it2) for _ in range(num_elements_to_verify)
    ]
    state_after_verification_restored = it2.get_state()

    # Verify data matches
    self.assertEqual(
        elements_after_checkpoint_original,
        elements_after_checkpoint_restored,
        msg=(
            "Data mismatch after checkpoint restore! This indicates the"
            " window_index bug."
        ),
    )

    # Verify window_index matches
    self.assertEqual(
        state_after_verification_original["window_index"],
        state_after_verification_restored["window_index"],
        msg=(
            "window_index mismatch: "
            f"original={state_after_verification_original['window_index']}, "
            f"restored={state_after_verification_restored['window_index']}. "
            "This indicates the checkpoint was not set properly."
        ),
    )

  def test_reverse_option(self):
    ds = dataset.MapDataset.range(12)
    using_map = shuffle.WindowShuffleMapDataset(ds, window_size=3, seed=42)
    using_iter_default = shuffle.WindowShuffleIterDataset(
        ds.to_iter_dataset(), window_size=3, seed=42
    )
    using_iter_reverse_true = shuffle.WindowShuffleIterDataset(
        ds.to_iter_dataset(), window_size=3, seed=42, reverse=True
    )
    using_iter_reverse_false = shuffle.WindowShuffleIterDataset(
        ds.to_iter_dataset(), window_size=3, seed=42, reverse=False
    )

    self.assertEqual(list(using_iter_default), list(using_iter_reverse_true))
    self.assertEqual(list(using_map), list(using_iter_reverse_false))
    self.assertEqual(list(using_map), [0, 1, 2, 5, 3, 4, 6, 7, 8, 11, 9, 10])
    self.assertEqual(
        list(using_iter_reverse_true), [2, 1, 0, 4, 3, 5, 8, 7, 6, 10, 9, 11]
    )

  def test_reverse_false_checkpoints(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=4, seed=42, reverse=False
    )
    ds_iter = ds.__iter__()
    checkpoints = []
    checkpoints_values = []
    for _ in range(self._DATASET_SIZE):
      checkpoints.append(ds_iter.get_state())
      checkpoints_values.append(next(ds_iter))
    for i in range(len(checkpoints)):
      ds_iter.set_state(checkpoints[i])
      self.assertDictEqual(
          ds_iter.get_state(),
          checkpoints[i],
          msg=f"Checkpoint state failed at {i}.",
      )
      values = list(ds_iter)
      self.assertEqual(
          checkpoints_values[i:],
          values,
          msg=f"Checkpoint values failed from checkpoint {i}.",
      )

  def test_reverse_false_checkpoint_restore_on_fresh_iterator(self):
    window_size = 10
    num_elements = 1000
    seed = 42
    num_full_windows_before_checkpoint = 5
    num_elements_to_verify = 20

    checkpoint_position = (
        num_full_windows_before_checkpoint * window_size + window_size // 2
    )

    ds = dataset.MapDataset.range(num_elements).to_iter_dataset()
    ds = shuffle.WindowShuffleIterDataset(
        ds, window_size=window_size, seed=seed, reverse=False
    )

    it1 = ds.__iter__()
    for _ in range(checkpoint_position):
      next(it1)
    checkpoint_state = it1.get_state()

    elements_after_checkpoint_original = [
        next(it1) for _ in range(num_elements_to_verify)
    ]
    state_after_verification_original = it1.get_state()

    ds2 = dataset.MapDataset.range(num_elements).to_iter_dataset()
    ds2 = shuffle.WindowShuffleIterDataset(
        ds2, window_size=window_size, seed=seed, reverse=False
    )
    it2 = ds2.__iter__()

    it2.set_state(checkpoint_state)

    elements_after_checkpoint_restored = [
        next(it2) for _ in range(num_elements_to_verify)
    ]
    state_after_verification_restored = it2.get_state()

    self.assertEqual(
        elements_after_checkpoint_original,
        elements_after_checkpoint_restored,
    )
    self.assertEqual(
        state_after_verification_original["window_index"],
        state_after_verification_restored["window_index"],
    )

  # Tests that the checkpoint regarding reverse flag in WindowShuffleIterDataset
  # follows the write-time configuration for backward-compatibility.
  #
  # Write \ Read          | Unset           | True            | False
  # ----------------------|-----------------|-----------------|----------------
  # Unset                 | True (Reverse)  | True (Reverse)  | Error
  # True                  | True (Reverse)  | True (Reverse)  | Error
  # False                 | Error           | Error           | False (Forward)
  # Legacy (Missing Key)  | True (Reverse)  | True (Reverse)  | Error
  @parameterized.named_parameters(
      dict(
          testcase_name="write_none_read_none",
          legacy_strip_key=False,
          write_reverse=None,
          read_reverse=None,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_none_read_true",
          legacy_strip_key=False,
          write_reverse=None,
          read_reverse=True,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_none_read_false",
          legacy_strip_key=False,
          write_reverse=None,
          read_reverse=False,
          expected_reverse=ValueError,
      ),
      dict(
          testcase_name="write_true_read_none",
          legacy_strip_key=False,
          write_reverse=True,
          read_reverse=None,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_true_read_true",
          legacy_strip_key=False,
          write_reverse=True,
          read_reverse=True,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_true_read_false",
          legacy_strip_key=False,
          write_reverse=True,
          read_reverse=False,
          expected_reverse=ValueError,
      ),
      dict(
          testcase_name="write_false_read_none",
          legacy_strip_key=False,
          write_reverse=False,
          read_reverse=None,
          expected_reverse=ValueError,
      ),
      dict(
          testcase_name="write_false_read_true",
          legacy_strip_key=False,
          write_reverse=False,
          read_reverse=True,
          expected_reverse=ValueError,
      ),
      dict(
          testcase_name="write_false_read_false",
          legacy_strip_key=False,
          write_reverse=False,
          read_reverse=False,
          expected_reverse=False,
      ),
      dict(
          testcase_name="write_legacy_read_none",
          legacy_strip_key=True,
          write_reverse=None,
          read_reverse=None,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_legacy_read_true",
          legacy_strip_key=True,
          write_reverse=None,
          read_reverse=True,
          expected_reverse=True,
      ),
      dict(
          testcase_name="write_legacy_read_false",
          legacy_strip_key=True,
          write_reverse=None,
          read_reverse=False,
          expected_reverse=ValueError,
      ),
  )
  def test_window_shuffle_iter_dataset_checkpoint_compatibility_matrix(
      self,
      legacy_strip_key: bool,
      write_reverse: bool | None,
      read_reverse: bool | None,
      expected_reverse: bool | type[ValueError],
  ):
    window_size = 10
    num_elements = 100
    seed = 42
    checkpoint_pos = 25

    # 1. Write checkpoint from dataset configured with write_reverse
    ds_write = dataset.MapDataset.range(num_elements).to_iter_dataset()
    kwargs_write = dict(window_size=window_size, seed=seed)
    if write_reverse is not None:
      kwargs_write["reverse"] = write_reverse
    ds_write = shuffle.WindowShuffleIterDataset(ds_write, **kwargs_write)
    it_write = iter(ds_write)
    for _ in range(checkpoint_pos):
      next(it_write)
    checkpoint_state = it_write.get_state()

    if legacy_strip_key:
      checkpoint_state.pop("reverse", None)

    # 2. Read / restore checkpoint into dataset configured with read_reverse
    ds_read = dataset.MapDataset.range(num_elements).to_iter_dataset()
    kwargs_read = dict(window_size=window_size, seed=seed)
    if read_reverse is not None:
      kwargs_read["reverse"] = read_reverse
    ds_read = shuffle.WindowShuffleIterDataset(ds_read, **kwargs_read)
    it_read = iter(ds_read)

    if expected_reverse is ValueError:
      with self.assertRaisesRegex(
          ValueError, "does not match dataset reverse configuration"
      ):
        it_read.set_state(checkpoint_state)
    else:
      assert isinstance(expected_reverse, bool)
      # Ground truth reference iterator with expected_reverse
      ref_ds = dataset.MapDataset.range(num_elements).to_iter_dataset()
      ref_ds = shuffle.WindowShuffleIterDataset(
          ref_ds, window_size=window_size, seed=seed, reverse=expected_reverse
      )
      ref_iter = iter(ref_ds)
      for _ in range(checkpoint_pos):
        next(ref_iter)
      expected_remaining = list(ref_iter)

      it_read.set_state(checkpoint_state)
      actual_remaining = list(it_read)
      self.assertEqual(actual_remaining, expected_remaining)

  def test_shuffled_raises_stop_iteration(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=self._WINDOW_SIZE, seed=42
    )
    ds_iter = ds.__iter__()
    with self.assertRaises(StopIteration):
      for _ in range(self._DATASET_SIZE + 1):
        next(ds_iter)

  def test_element_spec(self):
    ds = shuffle.WindowShuffleIterDataset(
        self.range_iter_ds, window_size=self._WINDOW_SIZE, seed=42
    )
    spec = dataset.get_element_spec(ds)
    self.assertEqual(spec.dtype, np.int64)
    self.assertEqual(spec.shape, ())


if __name__ == "__main__":
  absltest.main()
