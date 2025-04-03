from absl import flags
from absl.testing import absltest
from grain._src.python.dataset.sources import parquet_dataset
import grain.python as grain
import pyarrow as pa
import pyarrow.parquet as pq

flags.FLAGS.mark_as_parsed()

SOME_TEXT = [
    [
        "This is the first file the first record",
        "This is the first file the second record",
        "This is the first file the third record",
        "This is the first file the forth record",
    ],
    [
        "This is the second file the first record",
        "This is the second file the second record",
        "This is the second file the third record",
        "This is the second file the forth record",
    ],
]
INTERLEAVED_TEXT = [
    "This is the first file the first record",
    "This is the second file the first record",
    "This is the first file the second record",
    "This is the second file the second record",
    "This is the first file the third record",
    "This is the second file the third record",
    "This is the first file the forth record",
    "This is the second file the forth record",
]
WINDOWSHUFFLED_TEXT = [
    "This is the first file the second record",
    "This is the second file the first record",
    "This is the first file the first record",
    "This is the first file the third record",
    "This is the second file the second record",
    "This is the second file the third record",
    "This is the second file the forth record",
    "This is the first file the forth record",
]


class ParquetIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.filenames = []
    for i in range(len(SOME_TEXT)):
      temp_file = self.create_tempfile()
      filename = temp_file.full_path
      self.filenames.append(filename)
      table = pa.table({"text": SOME_TEXT[i]})
      pq.write_table(table, filename, row_group_size=2)

  def test_read_row_group(self):
    dataset = parquet_dataset.ParquetIterDataset(self.filenames[0])
    records = list(dataset)
    self.assertSequenceEqual(records, [{"text": x} for x in SOME_TEXT[0]])

  def test_checkpointing(self):
    dataset = parquet_dataset.ParquetIterDataset(self.filenames[0])
    grain.experimental.assert_equal_output_after_checkpoint(dataset)

  def test_sharded_files_and_interleaved_dataset(self):
    dataset = grain.MapDataset.source(self.filenames)
    dataset = dataset.map(parquet_dataset.ParquetIterDataset)
    dataset = grain.experimental.InterleaveIterDataset(
        dataset, cycle_length=len(self.filenames)
    )
    self.assertSequenceEqual(
        list(iter(dataset)), [{"text": x} for x in INTERLEAVED_TEXT]
    )

    dataset = grain.experimental.WindowShuffleIterDataset(
        dataset, window_size=3, seed=42
    )
    self.assertSequenceEqual(
        list(iter(dataset)), [{"text": x} for x in WINDOWSHUFFLED_TEXT]
    )


if __name__ == "__main__":
  absltest.main()
