# Copyright 2022 Google LLC
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
"""Data loaders create tf.data pipelines using DataSource and IndexSampler."""
import collections
import dataclasses
import functools
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
from etils import epath
from grain._src.core import sharding
from grain._src.core import usage_logging
from grain._src.core.config import config
import grain._src.core.constants as gc
from grain._src.tensorflow import batching
from grain._src.tensorflow import data_iterators
from grain._src.tensorflow import data_sources
from grain._src.tensorflow import index_dataset
from grain._src.tensorflow import transforms
from grain._src.tensorflow.ops import merge_by_min_value
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_RECORD_KEY_IN_MERGED_DATA_SOURCE = "_record_key_in_merged_data_source"
IteratorOptions = data_iterators.IteratorOptions
Transformations = transforms.Transformations


@dataclasses.dataclass
class _ParseTransform(transforms.MapTransform):
  """Transform for applying an optional parsing function."""

  parse_fn: Optional[data_sources.TfParseFn]

  def map(self, features):
    record = features.pop(gc.RECORD)
    if self.parse_fn:
      record = self.parse_fn(record)
    if isinstance(record, Mapping):
      return record | features
    features[gc.RECORD] = record
    return features


class TfDataLoader(collections.abc.Iterable):
  """Deterministic data loader for a single data source."""

  def __init__(
      self,
      *,
      source: data_sources.TfDataSource,
      sampler: index_dataset.TfIndexSampler,
      transformations: Transformations = (),
      batch_fn: Optional[batching.TfBatchFn] = None,
      iterator_options: Optional[IteratorOptions] = None,
      strict_transformations: bool = True,
      tf_data_options: Optional[tf.data.Options] = None,
      fewshot: bool = False,
      num_shots: Optional[int] = 1,
      fewshot_processors: Optional[Sequence[Any]] = None,
  ):
    """Initializes a new data loader.

    Args:
      source: The data source from which to read.
      sampler: The index sampler. This must emit record keys that are valid for
        the data source.
      transformations: Optional list of transformations to apply before
        batching.
      batch_fn: Function to use for batching the dataset. To disable batching
        pass `grain.TfBatchNone()`.
      iterator_options: Options passed to the data iterator.
      strict_transformations: If True (default) Grain will only apply valid
        transformations and fail if it cannot verify that a transformation is
        valid (see the rules for transformation in the documentation). For a
        transition period you can set this to False and Grain will apply all
        transformations. However this might make the input non-deterministic.
      tf_data_options: Options passed to tf.data.
      fewshot: whether data loader is setup for in context learning
      num_shots: number of shots/fewshot examples
      fewshot_processors: processors to modify fewshot examples.
    """
    usage_logging.log_event("TfDataLoader", tag_3="TfGrain")
    self.source = source
    self.sampler = sampler
    self._transformations = tuple(transformations)
    if batch_fn is not None:
      logging.warning(
          "Please pass the batching function in the list of transformations."
      )
      self._transformations += (batch_fn,)
    self._iterator_options = iterator_options
    self._strict_transformations = strict_transformations
    self._tf_data_options = tf_data_options

    self._fewshot = fewshot
    self._num_shots = num_shots or 1
    self._fewshot_processors = fewshot_processors or []

  def __iter__(self) -> data_iterators.TfGrainDatasetIterator:
    return data_iterators.TfGrainDatasetIterator(
        self,
        options=self._iterator_options,
        fewshot=self._fewshot,
        num_shots=self._num_shots,
        fewshot_processors=self._fewshot_processors,
    )

  def as_dataset(self, *, start_index: index_dataset.Index) -> tf.data.Dataset:
    """Returns a the tf.data input pipeline.

    If you want the pipeline to be reproducible you should at most use the
    following tf.data ops on the returned dataset: map/filter/batch
    Other operations, especially repeat/shuffle/cache/unbatch, will break with
    the reproducibility constraints.

    Args:
      start_index: The start_index where the input pipeline should start. This
        should be FIRST_INDEX when starting the pipeline from the beginning and
        NextIndex(last_seen_index) after the job got preempted. The user is
        responsible of keeping track of the last_seen_index.

    Returns:
      Returns the dataset.
    """
    index_ds = self.sampler.get_index_dataset(start_index)
    _validate_index_dataset(index_ds, require_dataset_index=False)
    ds = _map_index_dataset_using_data_source(self.source, index_ds)

    transformations = list(self._transformations)
    transformations.insert(0, _ParseTransform(self.source.get_parse_fn()))
    ds = transforms.apply_transformations(
        ds, transformations, strict=self._strict_transformations
    )
    if self._tf_data_options is not None:
      ds = ds.with_options(self._tf_data_options)
    return ds


def load_from_tfds(
    *,
    name: Optional[str] = None,
    split: str,
    data_dir: Optional[epath.PathLike] = None,
    tfds_info: Optional[tfds.core.DatasetInfo] = None,
    num_epochs: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[Any] = None,
    shard_options: sharding.ShardOptions,
    decoders: Optional[Any] = None,
    transformations: Transformations = (),
    batch_size: Optional[int] = None,
    batch_fn: Optional[batching.TfBatchFn] = None,
    tf_data_options: Optional[tf.data.Options] = None,
    cache_data_source: bool = False,
) -> TfDataLoader:
  """Create a data loader for a TFDS dataset.

  Name, split and data_dir are forwarded to TFDS. See the documentation there.

  Args:
    name: Name of the TFDS dataset. Provide either `name` (and `data_dir`) or
      `tfds_info`.
    split: Split of the dataset to use.
    data_dir: Optional data_dir in which to look for the dataset.
    tfds_info: Optional TFDS DatasetInfo object. Provide either `tfds_info` or
      `name` (and `data_dir` if applicable).
    num_epochs: See TfDefaultIndexSampler.
    shuffle: See TfDefaultIndexSampler.
    seed: See TfDefaultIndexSampler.
    shard_options: See TfDefaultIndexSampler.
    decoders: Additional decoding instructions for TFDS.
    transformations: List of local (stateless) transformations:
    batch_size: Optional batch size. If provided will apply TfBatch() with
      drop_remainder=False at the end.
    batch_fn: Custom batching function.
    tf_data_options: Options passed to tf.data.
    cache_data_source: Whether to cache the data source in memory.

  Returns:
    TfDataLoader for this dataset.
  """
  usage_logging.log_event("load_from_tfds", tag_3="TfGrain")
  if (name is None) == (tfds_info is None):
    raise ValueError("Please provide either `name` or `tfds_info`.")
  if name:
    source = data_sources.TfdsDataSource.from_name(
        name,
        data_dir=data_dir,
        split=split,
        decoders=decoders,
        cache=cache_data_source,
    )
  else:
    if data_dir:
      logging.error(
          "Ignoring data_dir in `load_from_tfds()` since `tfds_info` "
          "was provided."
      )
    source = data_sources.TfdsDataSource(
        dataset_info=tfds_info,
        split=split,
        decoders=decoders,
        cache=cache_data_source,
    )
  sampler = index_dataset.TfDefaultIndexSampler(
      num_records=len(source),
      shuffle=shuffle,
      seed=seed,
      num_epochs=num_epochs,
      shard_options=shard_options,
  )
  if batch_size is not None and batch_fn is not None:
    raise ValueError(
        "Arguments batch_size and batch_fn are mutually "
        "exclusive. Only use one of them."
    )
  if batch_size is not None:
    batch_fn = batching.TfBatch(batch_size, drop_remainder=num_epochs is None)
  return TfDataLoader(
      source=source,
      sampler=sampler,
      transformations=transformations,
      batch_fn=batch_fn,
      tf_data_options=tf_data_options,
  )


class TfMixtureDataLoader(collections.abc.Iterable):
  """Data loader for loading mixtures deterministically.

  Limitations:
    - Currently all data sources must be TfArrayRecordDataSources,
    TfBagDataSources, or TfInMemoryDatasources since we combine all sources into
    one source for better resource utilization.
    - The sampler needs to emit gc.DATASET_INDEX for each element.
  """

  def __init__(
      self,
      *,
      sources: Sequence[data_sources.TfDataSource],
      transformations_per_source: Sequence[Transformations],
      sampler: index_dataset.TfIndexSampler,
      transformations: Transformations = (),
      batch_fn: Optional[batching.TfBatchFn] = None,
      strict_transformations: bool = True,
      iterator_options: Optional[IteratorOptions] = None,
      tf_data_options: Optional[tf.data.Options] = None,
      prefetch_buffer_size: int = tf.data.AUTOTUNE,
  ):
    """Initializes a new data loader.

    Args:
      sources: The data sources from which to read.
      transformations_per_source:
      sampler: The index sampler. This must emit a dataset index and valid
        records within the dataset.
      transformations: Optional list of transformations to apply before
        batching.
      batch_fn: Deprecated. Function to use for batching the dataset. Please add
        the batching transformation to the list of transformations.
      strict_transformations: See TfDataLoader.
      iterator_options: Options passed to the data iterator.
      tf_data_options: Options passed to tf.data.
      prefetch_buffer_size: Size of prefetch buffer, if any. Autotuned if not
        specified.
    """
    usage_logging.log_event("TfMixtureDataLoader", tag_3="TfGrain")
    assert len(sources) == len(transformations_per_source)

    # Combine all data sources into one.
    if all(isinstance(s, data_sources.TfInMemoryDataSource) for s in sources):
      self.source = data_sources.TfInMemoryDataSource(
          values=tf.concat([s.values for s in sources], axis=0),  # pytype: disable=attribute-error
          parse_fn=None,  # Handled in transformpations per source.
      )
    else:
      # Combine data sources into a TfArrayRecordDataSource or TfBagDataSource.
      array_record_paths = []
      bag_paths = []
      for s in sources:
        if isinstance(s, data_sources.TfdsDataSource):
          s = s._source

        if isinstance(s, data_sources.TfArrayRecordDataSource):
          array_record_paths.extend(s._paths)
        elif isinstance(s, data_sources.TfBagDataSource):
          bag_paths.extend(s._paths)
        else:
          raise ValueError(f"Data source {s} is not yet supported in mixtures.")

      if array_record_paths and bag_paths:
        raise ValueError(
            "Please only mix datasets of the same format (currently ArrayRecord"
            " or Bag only)."
        )

      if not array_record_paths and not bag_paths:
        raise ValueError("No datasets provided.")

      if array_record_paths:
        self.source = data_sources.TfArrayRecordDataSource(array_record_paths)
      else:
        self.source = data_sources.TfBagDataSource(bag_paths)

    self.sampler = sampler
    self._records_per_dataset = [len(s) for s in sources]

    self._transformations_per_source = [
        list(ts) for ts in transformations_per_source
    ]
    for i, source in enumerate(sources):
      if parse_fn := source.get_parse_fn():
        self._transformations_per_source[i].insert(0, _ParseTransform(parse_fn))
    self._dataset_index_to_group, self._group_to_transformation = (
        self._create_groups(self._transformations_per_source)
    )

    self._transformations = tuple(transformations)
    if batch_fn is not None:
      logging.warning(
          "Please pass the batching function in the list of transformations."
      )
      self._transformations += (batch_fn,)
    self._iterator_options = iterator_options
    self._strict_transformations = strict_transformations
    self._tf_data_options = tf_data_options
    self._prefetch_buffer_size = prefetch_buffer_size

  def __iter__(self):
    return data_iterators.TfGrainDatasetIterator(
        self, options=self._iterator_options
    )

  @property
  def _num_datasets(self):
    return len(self._dataset_index_to_group)

  @property
  def _num_groups(self):
    return max(self._dataset_index_to_group) + 1

  def _create_groups(
      self, transformations_per_source: Sequence[Transformations]
  ) -> Tuple[Sequence[int], Sequence[Transformations]]:
    """Groups sources with the same transformations into groups."""
    key_to_group = {}
    dataset_index_to_group = []
    group_to_transformations = []
    for transformations in transformations_per_source:
      # TODO(mrit): Is this always safe? Should we be using __hash__()?
      key = str(transformations)
      if key not in key_to_group:
        key_to_group[key] = len(key_to_group)
        group_to_transformations.append(transformations)
      dataset_index_to_group.append(key_to_group[key])
    return dataset_index_to_group, group_to_transformations

  def as_dataset(self, *, start_index: index_dataset.Index) -> tf.data.Dataset:
    """Returns a the tf.data input pipeline.

    If you want the pipeline to be reproducible you should at most use the
    following tf.data ops on the returned dataset: map/filter/batch
    Other operations, especially repeat/shuffle/cache/unbatch, will break with
    the reproducibility constraints.

    Args:
      start_index: The start_index where the input pipeline should start. This
        should be FIRST_INDEX when starting the pipeline from the beginning and
        NextIndex(last_seen_index) after the job got preempted. The user is
        responsible of keeping track of the last_seen_index.

    Returns:
      Returns the dataset.
    """
    index_ds = self.sampler.get_index_dataset(start_index)
    _validate_index_dataset(index_ds, require_dataset_index=True)
    # We read all datasets through a single data source and need to offset
    # record_key's.
    index_ds = _add_global_record_key(
        index_ds,
        records_per_dataset=self._records_per_dataset,
        output_key=_RECORD_KEY_IN_MERGED_DATA_SOURCE,
    )
    ds = _map_index_dataset_using_data_source(
        self.source,
        index_ds,
        input_key=_RECORD_KEY_IN_MERGED_DATA_SOURCE,
        drop_input_key=True,
        prefetch_buffer_size=self._prefetch_buffer_size,
    )

    just_map_transforms = True
    for g in range(self._num_groups):
      for t in self._group_to_transformation[g]:
        if isinstance(t, transforms.MapTransform) or isinstance(
            t, transforms.RandomMapTransform
        ):
          continue
        just_map_transforms = False
        break
      if not just_map_transforms:
        break

    if just_map_transforms:
      ds = transforms.apply_transformations_by_dataset_index(
          ds, self._transformations_per_source
      )
      ds = transforms.apply_transformations(
          ds, self._transformations, strict=self._strict_transformations
      )
      if self._tf_data_options is not None:
        ds = ds.with_options(self._tf_data_options)
      return ds

    # Convert Sequence[int] to tensor for lookups in TF.
    task_index_to_group = tf.constant(
        self._dataset_index_to_group, dtype=tf.int64
    )

    def is_group(x: Mapping[str, Any], group: int):
      return task_index_to_group[x[gc.DATASET_INDEX]] == group

    def transform_group(ds: tf.data.Dataset, *, group: int) -> tf.data.Dataset:
      return transforms.apply_transformations(
          ds,
          self._group_to_transformation[group],
          strict=self._strict_transformations,
      )

    if self._num_groups == 1:
      # All data sources in the mixture have the same transformations. Apply
      # them and we are done.
      ds = transform_group(ds, group=0)
      ds = transforms.apply_transformations(ds, self._transformations)
      if self._tf_data_options is not None:
        ds = ds.with_options(self._tf_data_options)
      return ds

    # We treat each set of tasks with the same preprocessors as a group and
    # apply the preprocessors per group separately before merging the dataset
    # again. Construct `Dataset` for each group and merge back together later.

    dataset_per_group = []
    # SeqIO mixture only keep the output features of the first tasks. We keep
    # the output features that are shared by all tasks.
    common_features = None
    for group in range(self._num_groups):
      # Only process elements in this group.
      group_ds = ds.filter(functools.partial(is_group, group=group))
      group_ds = transform_group(group_ds, group=group)
      if common_features is None:
        common_features = set(group_ds.element_spec)
      else:
        common_features &= set(group_ds.element_spec)
      dataset_per_group.append(group_ds)

    # Drop features not present in all tasks.
    for i in range(self._num_groups):
      dataset_per_group[i] = dataset_per_group[i].map(
          lambda x: {k: v for k, v in x.items() if k in common_features},
          num_parallel_calls=tf.data.AUTOTUNE,
      )

    ds = merge_by_min_value.MergeByMinValueDataset(
        dataset_per_group, merge_field=gc.INDEX
    )
    ds = transforms.apply_transformations(
        ds, self._transformations, strict=self._strict_transformations
    )
    if self._tf_data_options is not None:
      ds = ds.with_options(self._tf_data_options)
    return ds


def _add_global_record_key(
    index_ds: tf.data.Dataset,
    *,
    records_per_dataset: Sequence[int],
    output_key: str,
) -> tf.data.Dataset:
  """Adds the feature `output_key` for the global record key.

  Args:
    index_ds:
    records_per_dataset:
    output_key:

  Returns:
    The new dataset with the added feature for the global record key.
  """
  offsets = np.concatenate(([0], np.cumsum(records_per_dataset[:-1])))
  offsets = tf.constant(offsets, dtype=tf.int64)
  records_per_dataset = tf.constant(records_per_dataset, dtype=tf.int64)

  def map_fn(features):
    dataset_index = features[gc.DATASET_INDEX]
    tf.debugging.assert_less(
        features[gc.RECORD_KEY], records_per_dataset[dataset_index]
    )
    features[output_key] = features[gc.RECORD_KEY] + offsets[dataset_index]
    return features

  return index_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)


def _validate_index_dataset(
    index_ds: tf.data.Dataset, *, require_dataset_index: bool
):
  """Raises an error if `index_ds` is not a valid index dataset.

  Valid index datasets are tf.data.Dataset objects which elements are
  dictionaries to tensors. The dictionaries must contain keys `gc.INDEX` and
  `gc.RECORD_KEY` (and sometimes gc.DATASET_INDEX).
  The dictionaries can contain additional keys.

  Args:
    index_ds: Index dataset to validate.
    require_dataset_index: Whether the index dataset also needs to contain
      gc.DATASET_INDEX.
  """
  if not isinstance(index_ds, tf.data.Dataset):
    raise ValueError(
        "IndexSampler.get_index_dataset() must return a tf.data.Dataset but "
        f"got {type(index_ds)}."
    )
  if not isinstance(index_ds.element_spec, dict):
    raise ValueError(
        "Index dataset elements must be dictionarios but got "
        f"{index_ds.element_spec}."
    )
  mandatory_keys = (gc.INDEX, gc.RECORD_KEY)
  if require_dataset_index:
    mandatory_keys += (gc.DATASET_INDEX,)
  if missing_keys := set(mandatory_keys) - set(index_ds.element_spec):
    raise ValueError(
        f"Index dataset is missing keys {missing_keys}. Index datasets must "
        f"contain keys {mandatory_keys}."
    )
  for key in mandatory_keys:
    actual_dtype = index_ds.element_spec[key].dtype
    actual_shape = index_ds.element_spec[key].shape
    if actual_dtype != tf.int64 or len(actual_shape):
      raise ValueError(
          f"Index dataset must contain feature {key} with dtype tf.int64 (got "
          f"{actual_dtype}) and shape (,) (got {actual_shape})."
      )


def _map_index_dataset_using_data_source(
    source: data_sources.TfDataSource,
    index_ds: tf.data.Dataset,
    *,
    input_key: str = gc.RECORD_KEY,
    output_key: str = gc.RECORD,
    drop_input_key: bool = False,
    prefetch_buffer_size: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
  """Returns a dataset with the records matching for the provided keys.

  Args:
    source: Data source from which to read.
    index_ds: The dataset with the keys. Elements must be dictionaries
      containing the keys to lookup in `input_key`.
    input_key: The name of the feature holding the record keys to look up.
    output_key: The name of the feature where the record values should be
      stored.
    drop_input_key: Whether to drop the features in `input_key`.
    prefetch_buffer_size: Size of prefetch buffer, if any. Autotuned if None.

  Returns:
    A dataset of the same cardinality as `index_ds`. The `output_key`
    contains the value stored in the array record files.
  """
  if input_key not in index_ds.element_spec:
    raise ValueError(
        f"Feature {input_key} not in input dictionary, available features: "
        f"{list(index_ds.element_spec)}."
    )
  if output_key in index_ds.element_spec:
    raise ValueError(
        f"Feature {output_key} is already present in input dictionary."
    )

  def lookup_fn(features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    features[output_key] = source[features[input_key]]
    if drop_input_key:
      del features[input_key]
    return features

  if config.tf_lookup_batch_size == 1:
    dataset = index_ds.map(
        lookup_fn, num_parallel_calls=config.tf_lookup_num_parallel_calls
    )
    return dataset.prefetch(prefetch_buffer_size)

  if config.tf_lookup_fast_warmup:
    # Split the first batch into a separate dataset. Will be concatenated at
    # the end again.
    warmup_index_ds = index_ds.take(config.tf_lookup_batch_size)
    index_ds = index_ds.skip(config.tf_lookup_batch_size)

    # We will make 10 calls with smaller batches and only 2 in parallel.
    warmup_batch_size = max(1, config.tf_lookup_batch_size // 10)
    warmup_dataset_cardinality = warmup_index_ds.cardinality()
    warmup_dataset = (
        warmup_index_ds.batch(warmup_batch_size)
        .map(lookup_fn, num_parallel_calls=2)
        .unbatch()
        .apply(
            tf.data.experimental.assert_cardinality(warmup_dataset_cardinality)
        )
    )

  dataset_cardinality = index_ds.cardinality()
  dataset = (
      index_ds.batch(config.tf_lookup_batch_size)
      .map(lookup_fn, num_parallel_calls=config.tf_lookup_num_parallel_calls)
      .unbatch()
      .apply(tf.data.experimental.assert_cardinality(dataset_cardinality))
  )

  if config.tf_lookup_fast_warmup:
    dataset = warmup_dataset.concatenate(dataset)

  return dataset.prefetch(prefetch_buffer_size)
