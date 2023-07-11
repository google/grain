"""Mock utils library for Grain."""

import contextlib
import sys
import types
from typing import Any

from grain._src.python import data_sources

try:
  import tensorflow_datasets as tfds  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
except ImportError:
  tfds: types.ModuleType = None


class FakeTfdsDataSource(data_sources.TfdsDataSource):
  """TfdsDataSource that returns fake data."""

  def __getitem__(self, record_key: int) -> Any:
    """This doesn't perform a lookup and just generates fake data.

    ### Miscellaneous
    * The examples are deterministically generated. "train" and "test" split
    will yield the same examples.

    Args:
      record_key: Index of the record to be looked up from the data source.

    Returns:
      A list of randomly generated records for each index.
    """
    if self._decoders:
      raise ValueError(
          "Using decoders is currently not supporting when "
          "mocking TfdsDataSource."
      )
    generator = tfds.testing.mocking.RandomFakeGenerator(self._features, 0)
    if record_key < 0 or record_key >= self._len:
      raise ValueError("Record key should be in [0, num_records)")
    return generator[record_key]


@contextlib.contextmanager
def mock_tfds_data_source():
  """Mocks PyGrain TfdsDataSource to generate random data.

  ### Usage example

  ```py
  import grain.python as grain

  dataset_name = ...
  grain.testing.mock_tfds_data_source():
    data_source = grain.TfdsDataSource.from_name(name=dataset_name,
        split="train")

    elements = data_source[1, 2, 3]

    for ex in elements:  # data_source will yield randomly generated examples.
      ex
  ```

  Yields:
    Scope in which TfdsDataSource is replaced with FakeTfdsDataSource.
  """
  # We cannot use mocking because the mocking context is not transferred to
  # child processes.
  if "grain.python" not in sys.modules:
    raise ValueError("grain.python module not found in sys.modules.")
  with tfds.testing.mock_data():
    sys.modules["grain.python"].TfdsDataSource = FakeTfdsDataSource
    try:
      yield
    finally:
      sys.modules["grain.python"].TfdsDataSource = data_sources.TfdsDataSource
