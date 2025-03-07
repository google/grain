from absl import flags
from absl.testing import absltest
from grain._src.python.dataset import parquet_dataset
import pyarrow as pa
import pyarrow.parquet as pq

flags.FLAGS.mark_as_parsed()

SOME_TEXT = [
    "This is the first record",
    "This is the second record",
    "This is the third record",
    "This is the forth record",
]


class ParquetIterDatasetTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    temp_file = self.create_tempfile()
    self.filename = temp_file.full_path
    table = pa.table({"text": SOME_TEXT})
    pq.write_table(table, self.filename, row_group_size=2)

  def test_read_row_group(self):
    dataset = parquet_dataset.ParquetIterDataset(self.filename)
    records = list(dataset)
    self.assertSequenceEqual(records, [{"text": x} for x in SOME_TEXT])

  def test_checkpointing(self):
    dataset = parquet_dataset.ParquetIterDataset(self.filename)
    it = dataset.__iter__()
    max_steps = len(SOME_TEXT)
    values_without_interruption = []
    checkpoints = []

    for _ in range(max_steps):
      checkpoints.append(it.get_state())
      values_without_interruption.append(next(it))

    for starting_step in range(max_steps):
      it.set_state(checkpoints[starting_step])
      for i in range(starting_step, max_steps):
        self.assertEqual(next(it), values_without_interruption[i])


if __name__ == "__main__":
  absltest.main()
