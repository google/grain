"""Setup.py file for grain.

Most project configs are in `pyproject.toml` -- prefer to modify
`pyproject.toml` over this file if possible.

There are extra building steps in this script:
  1. Generates proto bindings.
  2. Compiles C++ extensions using pybind11.
In case those are pre-built (e.g. when building wheels), you can skip this
extra build using `GRAIN_SKIP_EXTRA_BUILD=1`.
"""

import importlib.resources
import os
import setuptools
from setuptools import dist
from setuptools.command import build_py


class BinaryDistribution(dist.Distribution):
  """This class makes 'bdist_wheel' include an ABI tag on the wheel."""

  def has_ext_modules(self):
    return True


# If GRAIN_SKIP_EXTRA_BUILD=1, we are packaging pre-built binaries (e.g. in
# build_whl.sh). We can skip all build logic (C++ compilation and Proto
# generation).
if os.environ.get("GRAIN_SKIP_EXTRA_BUILD", "0") == "1":
  setuptools.setup(
      distclass=BinaryDistribution,
  )
else:
  # Full build logic for installing from source

  class GenerateProtosCommand(setuptools.Command):
    """Command to generate Python protobuf bindings."""

    description = "Generate Python protobuf bindings"
    user_options = []

    def initialize_options(self):
      pass

    def finalize_options(self):
      pass

    def run(self):
      from grpc_tools import protoc  # pylint: disable=g-import-not-at-top

      root_dir = os.path.dirname(os.path.abspath(__file__))

      proto_file = os.path.join(
          root_dir, "grain", "proto", "execution_summary.proto"
      )

      grpc_protos_include = str(
          importlib.resources.files("grpc_tools").joinpath("_proto")
      )

      proto_args = [
          "grpc_tools.protoc",
          f"--proto_path={grpc_protos_include}",
          f"--proto_path={root_dir}",
          f"--python_out={root_dir}",
          proto_file,
      ]

      if protoc.main(proto_args) != 0:
        raise RuntimeError(f"Error compiling proto: {proto_args}")

  class BuildPyCommand(build_py.build_py):
    """Run proto generation before build_py."""

    def run(self):
      self.run_command("generate_protos")
      super().run()

  from pybind11.setup_helpers import Pybind11Extension  # pylint: disable=g-import-not-at-top

  ext_modules = [
      Pybind11Extension(
          name="grain._src.python.experimental.index_shuffle.python.index_shuffle_module",
          sources=[
              "grain/_src/python/experimental/index_shuffle/python/index_shuffle_module.cc",
              "grain/_src/python/experimental/index_shuffle/index_shuffle.cc",
          ],
          include_dirs=["."],
      )
  ]

  setuptools.setup(
      distclass=BinaryDistribution,
      cmdclass={
          "build_py": BuildPyCommand,
          "generate_protos": GenerateProtosCommand,
      },
      ext_modules=ext_modules,
      setup_requires=["grpcio-tools"],
  )
