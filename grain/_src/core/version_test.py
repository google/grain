"""Tests to check that tests in OSS are run with the correct version of Python."""

import os
import sys
from absl.testing import absltest


class VersionTest(absltest.TestCase):

  def test_python_version(self):
    expected = os.getenv("PYTHON_VERSION")
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != expected:
      raise ValueError(
          f"expected version '{expected}' is different than returned"
          f" '{current}'"
      )


if __name__ == "__main__":
  absltest.main()
