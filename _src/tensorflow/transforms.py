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
"""Interfaces for transformations.

Grain distinguishes 3 categories of transformations:
1) Local transformations (filter, map). This must be stateless and should be
   implemented as subclasses of MapTransform, RandomMapTransform or
   FilterTransform (this helps Grain to recognize these).
2) Valid global transformations provided by Grain (e.g. batch, pack, cache).
   Global transformations can easily violate the rules for transformations and
   we recommend users to only use the set of global transformations provided by
   Grain.
3) Unsafe global transformations provided by the user (e.g. repeat, unbatch).
   These can be valid but Grain has no option of verifying that. We don't
   recommend using these and usually there are better solutions (e.g. use
   `num_epochs` in index sampler instead of repeat()).
"""
from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Mapping, Sequence, TypeVar, Union

from absl import logging
from clu import preprocess_spec
from grain._src.core import constants
import tensorflow as tf
from typing_extensions import final

# Below we define the types for TfGrain. PyGrain will have similar types:
# Element -> PyTree[Any].
# TfSeed -> np.random.Generator
# TfBool -> bool
# In the future we might work out how to use the same classes for both sets of
# types.

# Anything can be a PyTree (it's either a container or leaf). We define
# PyTree[T] as a PyTree where all leaves are of type T.
# See https://jax.readthedocs.io/en/latest/pytrees.html.
L = TypeVar("L")  # pylint: disable=invalid-name
PyTree = Union[L, Sequence["PyTree[L]"], Mapping[str, "PyTree[L]"]]

# For tf.data elements need to be tensors.
TfTensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Element = PyTree[TfTensor]
# Tensor for stateless random seed. Should have shape [2] and dtype int32 or
# int64.
TfSeed = tf.Tensor
# Scalar bool tensor.
TfBool = tf.Tensor


class MapTransform(abc.ABC):
  """Abstract base class for all 1:1 transformations of elements."""

  @abc.abstractmethod
  def map(self, element: Element) -> Element:
    """Maps a single element."""


class RandomMapTransform(abc.ABC):
  """Abstract base class for all random 1:1 transformations of elements."""

  @abc.abstractmethod
  def random_map(self, element: Element, rng: TfSeed) -> Element:
    """Maps a single element."""


class FilterTransform(abc.ABC):
  """Abstract base class for filter operations for individual elements."""

  @abc.abstractmethod
  def filter(self, element: Element) -> TfBool:
    """Filters a single element."""


class GlobalTfDataTransform(abc.ABC):
  """Base class for global transformations."""
  # This is not a public API since it would allow transformations that violate
  # Grain's restriction and guarantees.

  @abc.abstractmethod
  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Applies this transformation to the dataset."""


class UnsafeTfDataTransform(GlobalTfDataTransform):
  """Use this base class to implement your own tf.data transformation.

  WARNING:
  This should be your last resort. By subclassing this transform you get access
  to the underlying `tf.data.Dataset` object and can modify it freely. Grain
  will apply your transform as is but it cannot verify that your new dataset is
  deterministic and follows the the rules for transformations. Your input
  pipeline will run fine but most likely Grain will not recover from
  checkpoints correctly and skip some elements (or worse).
  That said, your use might be valid and we recommend to reach out to us to
  check if using UnsafeTfDataTransform is your best option.

  ```
  class DummyTransform(grain.UnsafeTfDataTransform):

    def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
      del dataset
      return tf.data.experimental.from_list(["I", "tricked", "Grain"])
  ```
  """


@final
@dataclasses.dataclass(frozen=True)
class CacheTransform(GlobalTfDataTransform):
  """Caches the whole tf.data.Dataset in memory.

  Use carefully! This can use a lot of memory depending on the size of the
  transformed dataset and Grain cannot guarantee that this will not cause an OOM
  errors.
  We only recommend this transformation in one case:
  Your model is input bound during evaluation but your evaluation data fits into
  in the CPU memory. In this case you can place this transformation in your
  list of transformations (likely at the end) and get the DatasetIterator and
  use it for multiple evaluations.

  If you want to cache the data loaded from disk (before the transformations)
  you should cache in the data source (if supported) or TfInMemoryDatasource.
  """

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
      raise ValueError("Cannot cache infinite dataset in memory.")
    return dataset.cache()


@final
@dataclasses.dataclass(frozen=True)
class IgnoreErrorsTransform(GlobalTfDataTransform):
  """Drops tf.data.Dataset elements that cause errors.

  For more info see:
  https://www.tensorflow.org/api_docs/python/tf/data/Dataset#ignore_errors

  Avoid using this if you can. This will silently drop any elements and is
  limitted to the tf.data backend. We won't support this transformation when
  running without tf.data!
  """

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.ignore_errors()


LocalTransform = Union[MapTransform, RandomMapTransform, FilterTransform,
                       preprocess_spec.PreprocessFn]
Transformation = Union[LocalTransform, GlobalTfDataTransform]
Transformations = Sequence[Transformation]


def _random_map_fn(features: Element, *,
                   transform: RandomMapTransform) -> Element:
  """Calls transform.random_map() with a random seed."""
  if not isinstance(features, dict):
    raise ValueError(
        "In Grain elements must be dictionaries but transformation "
        f"{transform} got {features}.")
  if constants.SEED not in features:
    raise ValueError(
        f"Random seed feature not present to apply {transform}. Either this "
        "was not created (by not passing a seed to the index sampler or a "
        "previous transformations removed it. Please do not remove Grain meta "
        "features (feature names starting with '_'.")
  next_seed, seed = tf.unstack(
      tf.random.experimental.stateless_split(features.pop(constants.SEED)))
  features = transform.random_map(features, seed)
  features[constants.SEED] = next_seed
  return features


def _try_apply_clu_preprocess_op(
    ds: tf.data.Dataset, preprocess_op) -> Union[tf.data.Dataset, Exception]:
  """Tries to apply a clu.preprocess_spec.PreprocessOp to the dataset.

  Args:
    ds: Dataset.
    preprocess_op: An object implementing the clu.preprocess_spec.PreprocessOp
      protocol.

  Returns:
    The transformed dataset if sucessful, an exception otherwise.
  """
  try:
    # clu.preprocess_spec.PreprocessOp gets called for a single element.
    ds = ds.map(preprocess_op, num_parallel_calls=tf.data.AUTOTUNE)
    assert isinstance(ds, tf.data.Dataset)
  except Exception as e:  # pylint: disable=broad-except
    return e
  logging.warning(
      "Applied deprecated PreprocessOp %s. Please subclass "
      "`grain.MapTransform.`", preprocess_op)
  return ds


def _try_apply_seqio_preprocessor(
    ds: tf.data.Dataset, preprocessor) -> Union[tf.data.Dataset, Exception]:
  """Tries to apply a SeqIO preprocessor to the dataset.

  Args:
    ds: Dataset.
    preprocessor: An object implementing the SeqIO preprocessor interface:
      Callable[[tf.data.Dataset], tf.data.Dataset].

  Returns:
    The transformed dataset if sucessful, an exception otherwise.
  """
  # In non-strict mode we allow arbitrary SeqIO preprocessors that could
  # break determinism. This smoothens the transition for SeqIO users who
  # have preprocessors that should be safe to use.
  # SeqIO preprocessor interface is
  # Callable[[tf.data.Dataset], tf.data.Dataset].
  try:
    ds = preprocessor(ds)
    assert isinstance(ds, tf.data.Dataset)
  except Exception as e:  # pylint: disable=broad-except
    return e
  logging.warning(
      "Applied potential unsafe preprocessor %s. Are you "
      "using plain SeqIO preprocessors?", preprocessor)
  return ds


def apply_transformations(ds: tf.data.Dataset,
                          transforms: Transformations,
                          *,
                          strict: bool = True) -> tf.data.Dataset:
  """Applies the transformations to the dataset.

  Args:
    ds: Input dataset.
    transforms: List of transformation that will be applied (in order) to the
      input dataset.
    strict: If False will allow transformations which don't conform with Grain's
      rule of transformations and could break determinism. Use carefully!

  Returns:
    The transformed dataset.
  """
  seed_required = any(isinstance(t, RandomMapTransform) for t in transforms)
  if seed_required and constants.SEED not in ds.element_spec:
    raise ValueError(
        "Cannot apply random transformations without a random seed feature. "
        "Passing a seed to the IndexSampler should create a SEED feature. List "
        f"of trainsformations: {transforms}.")

  for i, t in enumerate(transforms):
    if not isinstance(ds.element_spec, Mapping):
      raise ValueError(
          "Grain expects dataset elements to be dictionaries but got "
          f"{ds.element_spec}.")
    if isinstance(t, MapTransform):
      ds = ds.map(t.map, num_parallel_calls=tf.data.AUTOTUNE)
    elif isinstance(t, RandomMapTransform):
      map_fn = functools.partial(_random_map_fn, transform=t)
      ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    elif isinstance(t, FilterTransform):
      ds = ds.filter(t.filter)
    elif isinstance(t, GlobalTfDataTransform):
      ds = t.apply_to_dataset(ds)
    elif isinstance(t, preprocess_spec.PreprocessFn):
      logging.warning(
          "Applying deprecated PreprocessFn (%s). Please pass a sequence of "
          "`grain.MapTransform.`s instead.", t)
      ds = ds.map(t, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      ds_or_exception = _try_apply_clu_preprocess_op(ds, t)
      if isinstance(ds_or_exception, Exception) and not strict:
        ds_or_exception = _try_apply_seqio_preprocessor(ds, t)
      if isinstance(ds_or_exception, Exception):
        raise ValueError(f"Could not apply transform {t} to dataset {ds}."
                        ) from ds_or_exception
      ds = ds_or_exception
    if not isinstance(ds.element_spec, Mapping):
      raise ValueError(
          "Grain expects dataset elements to be dictionaries but got "
          f"{ds.element_spec} after transform {t}.")
    if constants.INDEX not in ds.element_spec:
      raise ValueError(
          f"Transform {t} removed the INDEX feature. Please do not remove "
          "features with keys in grain.META_FEATURES. Grain needs these for "
          "correctness.")
    if ds.element_spec[constants.INDEX].dtype != tf.int64:
      raise ValueError(
          f"Transform {t} changed the dtype of the INDEX feature to "
          f"{ds.element_spec[constants.INDEX].dtype}. Please do not change "
          "types of the features with keys in grain.META_FEATURES")
    seed_required = any(
        isinstance(t, RandomMapTransform) for t in transforms[i + 1:])
    if seed_required and constants.SEED not in ds.element_spec:
      raise ValueError(
          f"Transform {t} removed the SEED feature but a later transform needs "
          "it. Please do not remove features with keys in grain.META_FEATURES. "
          "Grain needs these for correctness.")
  return ds
