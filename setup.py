"""Setup.py file for grain.

Most project configs are in `pyproject.toml` -- prefer to modify
`pyproject.toml` over this file if possible.
"""

import setuptools
from setuptools import dist


class BinaryDistribution(dist.Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True


setuptools.setup(
    distclass=BinaryDistribution,
)
