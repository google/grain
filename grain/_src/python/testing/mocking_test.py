"""Tests for python.grain.mocking."""

from absl.testing import absltest
import grain.python as grain
import tensorflow_datasets as tfds


class MockingTest(absltest.TestCase):

  def test_mocking_wmt_translate_from_constructor(self):
    with grain.testing.mock_tfds_data_source():
      builder = tfds.builder(name="wmt17_translate/de-en")
      data_source = grain.TfdsDataSource(
          dataset_info=builder.info, split="train"
      )
      self.assertIsNotNone(data_source[42])

  def test_mocking_wmt_translate_from_name(self):
    dataset_name = "wmt17_translate/de-en"
    with grain.testing.mock_tfds_data_source():
      data_source = grain.TfdsDataSource.from_name(
          name=dataset_name, split="train"
      )
      self.assertIsNotNone(data_source[42])

  def test_mocking_negative_index(self):
    dataset_name = "wmt17_translate/de-en"
    with grain.testing.mock_tfds_data_source(), self.assertRaises(ValueError):
      data_source = grain.TfdsDataSource.from_name(
          name=dataset_name, split="train"
      )
      data_source[-1]  # pylint: disable=pointless-statement

  def test_mocking_large_index(self):
    dataset_name = "wmt17_translate/de-en"
    with grain.testing.mock_tfds_data_source(), self.assertRaises(ValueError):
      data_source = grain.TfdsDataSource.from_name(
          name=dataset_name, split="train"
      )
      data_source[10**7]  # pylint: disable=pointless-statement

  def test_mocking_deterministic_index_retrieval(self):
    dataset_name = "wmt17_translate/de-en"
    with grain.testing.mock_tfds_data_source():
      first_data_source = grain.TfdsDataSource.from_name(
          name=dataset_name, split="train"
      )
      second_data_source = grain.TfdsDataSource.from_name(
          name=dataset_name, split="train"
      )
      for i in range(3):
        # assert len(first_data_source[i]) == 0, list(first_data_source)
        self.assertEqual(first_data_source[i], second_data_source[i])


if __name__ == "__main__":
  absltest.main()
