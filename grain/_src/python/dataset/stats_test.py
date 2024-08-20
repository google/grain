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
"""Tests for stats.py."""
import contextlib
from typing import Sequence

from grain._src.python.dataset import stats

from absl.testing import absltest


class TestStats(stats.Stats):

  def __init__(self, name: str, parents: Sequence[stats.Stats]):
    super().__init__(name, parents)
    self.recorded_time = 0
    self.recorded_spec = False
    self.reported = False

  @contextlib.contextmanager
  def record_self_time(self):
    yield
    self.recorded_time += 1

  def record_output_spec(self, element):
    self.recorded_spec = True
    return element

  def report(self):
    self.reported = True


def _make_stats_tree(cls):
  return cls(
      "root",
      [
          cls(
              "left",
              [
                  cls("left_left", []),
                  cls("left_right", []),
              ],
          ),
          cls(
              "right",
              [
                  cls("right_left", []),
                  cls("right_right", []),
              ],
          ),
      ],
  )


def _for_each_node(fn, nodes):
  to_visit = list(nodes)
  while to_visit:
    node = to_visit.pop(0)
    fn(node)
    to_visit.extend(node._parents)


class StatsTest(absltest.TestCase):

  def test_correct_output_node(self):
    s = _make_stats_tree(TestStats)
    self.assertTrue(s._is_output)
    _for_each_node(lambda node: self.assertFalse(node._is_output), s._parents)

  def test_record_self_time(self):
    s = _make_stats_tree(TestStats)
    with s.record_self_time():
      pass
    self.assertEqual(s.recorded_time, 1)

  def test_record_output_spec(self):
    s = _make_stats_tree(TestStats)
    s.record_output_spec(1)
    self.assertTrue(s.recorded_spec)

  def test_report(self):
    s = _make_stats_tree(TestStats)
    s.report()
    self.assertTrue(s.reported)


class NoopStatsTest(absltest.TestCase):

  def test_record_self_time(self):
    s = _make_stats_tree(stats.NoopStats)
    with s.record_self_time():
      pass
    s = s._parents[0]
    with s.record_self_time():
      pass

  def test_record_output_spec(self):
    s = _make_stats_tree(stats.NoopStats)
    s.record_output_spec(1)
    s = s._parents[0]
    s.record_output_spec(1)

  def test_report(self):
    s = _make_stats_tree(stats.NoopStats)
    s.report()
    s = s._parents[0]
    s.report()


if __name__ == "__main__":
  absltest.main()
