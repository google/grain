"""Tests for sharding."""
from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import sharding


class ShardingTest(parameterized.TestCase):

  @parameterized.parameters(
      # num_examples, shard_index, shard_count, drop_remainder, expected_output.
      (9, 0, 1, True, (0, 9)),
      (9, 0, 2, True, (0, 4)),
      (9, 1, 2, True, (4, 8)),  # Last example gets dropped.
      (9, 0, 3, True, (0, 3)),
      (9, 1, 3, True, (3, 6)),
      (9, 2, 3, True, (6, 9)),
      (9, 0, 1, False, (0, 9)),
      (9, 0, 2, False, (0, 5)),  # First shard gets an extra example.
      (9, 1, 2, False, (5, 9)),
      (8, 0, 3, False, (0, 3)),  # First 2 shards get 1 example each.
      (8, 1, 3, False, (3, 6)),
      (8, 2, 3, False, (6, 8)),
  )
  def test_sharding(
      self,
      num_examples: int,
      shard_index: int,
      shard_count: int,
      drop_remainder,
      expected_output: tuple[int, int],
  ):
    shard_options = sharding.ShardOptions(
        shard_index=shard_index,
        shard_count=shard_count,
        drop_remainder=drop_remainder,
    )
    actual_output = sharding.even_split(num_examples, shard_options)
    self.assertEqual(actual_output, expected_output)


if __name__ == '__main__':
  absltest.main()
