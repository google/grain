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
"""Visualization tool for {Map|Iter}Dataset for debugging."""

from __future__ import annotations

import contextlib
import pprint
from typing import Any, Callable, Generator, TypeVar

from grain._src.core import tree_lib
from grain._src.python import options
from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import prefetch

T = TypeVar('T')

_EDGE_TEMPLATE = r"""{input_spec}
  ││
  ││  {transform}
  ││
  ╲╱
{output_spec}
"""


class _SpecTrackingMapDataset(dataset.MapDataset[T]):
  """A MapDataset that tracks spec of its parent outputs."""

  def __init__(
      self, parent: dataset.MapDataset, mock_source_output: bytes | None
  ):
    """Initializes the instance.

    Args:
      parent: The dataset to track.
      mock_source_output: Value to use as the output of the parent dataset if
        it's a source. If None, the actual source output will be used.
    """
    super().__init__(parent)
    if parent.parents:
      self._mock_output = None
    else:
      # This is a source dataset.
      self._mock_output = mock_source_output
    self.output_spec = None

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, idx):
    if self._mock_output is not None:
      result = self._mock_output
    else:
      result = self._parent.__getitem__(idx)
    self.output_spec = tree_lib.spec_like(result)
    return result

  def to_iter_dataset(
      self,
      read_options: options.ReadOptions | None = None,
      *,
      allow_nones: bool = False,
  ) -> dataset.IterDataset[T]:
    return _SpecTrackingIterDataset(
        super().to_iter_dataset(read_options, allow_nones=allow_nones),
        mock_source_output=None,
    )


class _SpecTrackingIterDataset(dataset.IterDataset[T]):
  """An IterDataset that tracks spec of its parent outputs."""

  def __init__(
      self, parent: dataset.IterDataset, mock_source_output: bytes | None
  ):
    """Initializes the instance.

    Args:
      parent: The dataset to track.
      mock_source_output: Value to use as the output of the parent dataset if
        it's a source. If None, the actual source output will be used.
    """
    super().__init__(parent)
    if parent.parents:
      self._mock_output = None
    else:
      # This is a source dataset.
      self._mock_output = mock_source_output
    self.output_spec = None

  def __iter__(self) -> dataset.DatasetIterator[T]:
    if isinstance(self._parent, prefetch.MultiprocessPrefetchIterDataset):
      # Visualization only fetches a single element -- prefetch will not be
      # useful, so we just skip it since it does not change the spec.
      parent_iter = self._parent._parent.__iter__()  # pylint: disable=protected-access
    else:
      parent_iter = self._parent.__iter__()
    return _SpecTrackingDatasetIterator(
        parent_iter,
        lambda spec: setattr(self, 'output_spec', spec),
        self._mock_output,
    )


class _SpecTrackingDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator that tracks spec of its parent iterator outputs."""

  def __init__(
      self,
      parent_iter: dataset.DatasetIterator[T],
      spec_update_fn: Callable[[Any], None],
      mock_output: bytes | None,
  ):
    """Initializes the instance.

    Args:
      parent_iter: Iterator to track.
      spec_update_fn: callback to update the spec of the iterator.
      mock_output: Value to use as the output of the parent iterator. If None,
        the actual iterator output will be used.
    """
    super().__init__(parent_iter)
    self._spec_update_fn = spec_update_fn
    self._mock_output = mock_output

  def __next__(self) -> T:
    result = self._mock_output or self._parent.__next__()
    self._spec_update_fn(tree_lib.spec_like(result))
    return result

  def set_state(self, state: dict[str, Any]) -> None:
    self._parent.set_state(state)

  def get_state(self) -> dict[str, Any]:
    return self._parent.get_state()


def _build_visualization_from_tracked_spec(
    ds: dataset.MapDataset | dataset.IterDataset,
) -> str:
  """Builds a visualization string from a dataset with specs already tracked."""
  assert isinstance(ds, (_SpecTrackingMapDataset, _SpecTrackingIterDataset)), ds
  assert len(ds.parents) == 1, ds.parents
  tracked_ds = ds.parents[0]
  if tracked_ds.parents:
    # This relies on the fact that tracking datasets are inserted after every
    # transformation.
    # We only visualize the first parent to avoid cluttering the visualization
    # with the entire tree. Note that this could be improved by e.g. showing
    # the first and the last parent branch. The most common multi-parent
    # transformations are mix and select_from_datasets. This also implicitly
    # relies on the fact that by fetching a single element we will touch the
    # first parent. This is true for `mix`, but is not necessarily true for
    # `select_from_datasets`.
    parent_vis = _build_visualization_from_tracked_spec(tracked_ds.parents[0])
    transform_repr = str(tracked_ds)
  else:
    # This dataset tracks the source output.
    parent_vis = str(tracked_ds)
    transform_repr = ''
  return _EDGE_TEMPLATE.format(
      input_spec=parent_vis,
      transform=transform_repr,
      output_spec=pprint.pformat(ds.output_spec),
  )


@contextlib.contextmanager
def _patch_dataset_with_spec_tracking(
    ds: dataset.MapDataset | dataset.IterDataset,
    mock_source_output: bytes | None,
) -> Generator[
    tuple[dataset.MapDataset | dataset.IterDataset, list[str]], None, None
]:
  """Inserts a spec tracking node after each transformation in a dataset.

  WARNING: Mutates the dataset in place. Implemented as a context manager to
  ensure that the dataset is restored to its original state after the tracking
  is done.

  Args:
    ds: The dataset to patch.
    mock_source_output: Value to use as the output of the source dataset. If
      None, the actual source output will be used.

  Yields:
    The patched dataset and a list of multi-parent dataset representations.
  """
  multiparent_datasets = []

  def _patch(ds):
    if len(ds.parents) > 1:
      multiparent_datasets.append(str(ds))
    ds._parents = [_patch(p) for p in ds.parents]  # pylint: disable=protected-access
    if isinstance(ds, dataset.MapDataset):
      return _SpecTrackingMapDataset(ds, mock_source_output)
    else:
      return _SpecTrackingIterDataset(ds, mock_source_output)

  def _unpatch(ds):
    assert isinstance(
        ds, (_SpecTrackingMapDataset, _SpecTrackingIterDataset)
    ), ds
    ds = ds.parents[0]
    assert not isinstance(
        ds, (_SpecTrackingMapDataset, _SpecTrackingIterDataset)
    ), ds
    ds._parents = [_unpatch(p) for p in ds.parents]  # pylint: disable=protected-access
    return ds

  ds = _patch(ds)
  try:
    yield ds, multiparent_datasets
  finally:
    _unpatch(ds)


def _build_visualization_str(
    ds: dataset.MapDataset | dataset.IterDataset,
    mock_source_output: bytes | None,
) -> str:
  """Fetches a single dataset element and builds a visualization string.

  Args:
    ds: The dataset to build visualization for.
    mock_source_output: Value to use as the output of the source dataset. If
      None, the actual source output will be used.

  Returns:
    The visualization string.
  """
  with _patch_dataset_with_spec_tracking(ds, mock_source_output) as (
      tracked_ds,
      multiparent_datasets,
  ):
    if isinstance(tracked_ds, dataset.MapDataset):
      _ = tracked_ds[0]
    else:
      _ = next(tracked_ds.__iter__())
    if multiparent_datasets:
      result = (
          'WARNING: Detected multi-parent datasets: '
          f'{", ".join(multiparent_datasets)}. Only displaying the first '
          'parent.\n\n'
      )
    else:
      result = ''
    return result + _build_visualization_from_tracked_spec(tracked_ds)


def visualize_dataset(
    ds: dataset.MapDataset | dataset.IterDataset,
    mock_source_output: bytes | None = None,
) -> None:
  """Produces a visualization of the dataset in stdout for debugging.

  WARNING: This function will fetch a single element from the dataset. Provide
  a mock source output to avoid touching the actual data.

  Do not rely on the specific produced string as it may be subject to changes.

  Args:
    ds: The dataset to visualize.
    mock_source_output: Value to use as the output of the source dataset. If
      None, the actual source output will be used.
  """
  print(_build_visualization_str(ds, mock_source_output))
