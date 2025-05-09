"""Provides an `IterDataset` for TFRecord file format."""

import codecs
import struct
from typing import BinaryIO, TypeVar

from grain._src.python.dataset import dataset


T = TypeVar("T")

# Format of a single tf_record:
#  uint64    length of record in bytes
#  uint32    masked crc of length
#  bytes     record data
#  uint32    masked crc of data
_UNIT32_SIZE_IN_BYTES = 4
_UNIT64_SIZE_IN_BYTES = 8


# TODO: b/416516712 - Move this to a common helper TFRecordReader class.
def _read_next_tf_record(reader: BinaryIO) -> bytes:
  """Reads the next record from the reader."""
  # Read the length and the length mask of the tf_record (uint64 and uint32
  # respectively)
  buf_length_expected = _UNIT64_SIZE_IN_BYTES + _UNIT32_SIZE_IN_BYTES
  buf = reader.read(buf_length_expected)
  if not buf:
    # If the buffer is empty, we have reached the end of the dataset.
    raise StopIteration()
  if len(buf) != buf_length_expected:
    raise ValueError(
        f"Not a valid TFRecord. Fewer than {buf_length_expected} bytes:"
        f" {codecs.encode(buf, 'hex')}"
    )
  length, _ = struct.unpack("<QI", buf)
  # TODO: b/412697846 - Add CRC check for length mask mismatch.

  # Read the data and the data mask of the tf_record (the length read earlier
  # and uint32 respectively)
  buf_length_expected = length + _UNIT32_SIZE_IN_BYTES
  buf = reader.read(buf_length_expected)
  if len(buf) != buf_length_expected:
    raise ValueError(
        f"Not a valid TFRecord. Fewer than {buf_length_expected} bytes:"
        f" {codecs.encode(buf, 'hex')}"
    )
  data, _ = struct.unpack("<%dsI" % length, buf)
  # TODO: b/412697846 - Add CRC check for data mask mismatch.
  return data


class _TFRecordDatasetIterator(dataset.DatasetIterator[T]):
  """A DatasetIterator for TFRecord file format."""

  def __init__(self, path: str):
    super().__init__()
    self._reader = open(path, "rb")

  def __next__(self) -> T:
    return _read_next_tf_record(self._reader)

  def get_state(self) -> dict[str, int]:
    return {
        "reader_offset": self._reader.tell(),
    }

  def set_state(self, state: dict[str, int]):
    self._reader.seek(state["reader_offset"])

  def __del__(self):
    self._reader.close()


class TFRecordIterDataset(dataset.IterDataset[T]):
  """An IterDataset for a TFRecord format file."""

  def __init__(self, path: str):
    super().__init__()
    self._path = path

  def __iter__(self) -> dataset.DatasetIterator[T]:
    return _TFRecordDatasetIterator[T](self._path)
