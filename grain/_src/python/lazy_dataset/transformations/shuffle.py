"""Implements global shuffle transformation."""

from typing import Optional, TypeVar

from grain._src.python.experimental.index_shuffle.python import index_shuffle_module as index_shuffle
from grain._src.python.lazy_dataset import lazy_dataset

T = TypeVar("T")


@lazy_dataset.lazy_map_dataset_function("shuffle")
class ShuffleLazyMapDataset(lazy_dataset.LazyMapDataset[T]):
  """Shuffles the parent dataset."""

  def __init__(
      self,
      parent: lazy_dataset.LazyMapDataset[T],
      *,
      reshuffle_each_epoch: bool = True,
      seed: int,
  ):
    super().__init__()
    self._parent = parent
    self._seed = seed
    self._reshuffle_each_epoch = reshuffle_each_epoch

  @property
  def sparse(self) -> bool:
    return self._parent.sparse

  def __len__(self) -> int:
    return len(self._parent)

  def __getitem__(self, index: int) -> Optional[T]:
    length = len(self._parent)
    epoch = index // length
    index_in_epoch = index % length
    if self._reshuffle_each_epoch:
      # index_shuffle expects 32-bit integers
      seed = (self._seed + epoch) % 2**32
    else:
      seed = self._seed
    shuffled_index_in_epoch = index_shuffle.index_shuffle(
        index_in_epoch, max_index=length - 1, seed=seed, rounds=4
    )
    shuffled_index = shuffled_index_in_epoch + epoch * length
    return self._parent[shuffled_index]
