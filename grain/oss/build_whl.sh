#!/bin/sh
# build wheel for python version specified in $PYTHON_VERSION

set -e -x

OUTPUT_DIR="${OUTPUT_DIR:-/tmp/grain}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

main() {
  # Remove .bazelrc if it already exists
  [ -e .bazelrc ] && rm .bazelrc

  # Only use __init__.py files that are present in the source. Without this,
  # Bazel will create empty __init__.py files for all subdirectories,
  # occasionally(!) overriding the source __init__.py files.
  write_to_bazelrc "build --incompatible_default_to_explicit_init_py"

  # Enable host OS specific configs. For instance, "build:linux" will be used
  # automatically when building on Linux.
  write_to_bazelrc "build --enable_platform_specific_config"
  # Bazel 7.0.0 no longer supports dynamic symbol lookup on macOS. To resolve
  # undefined symbol errors in macOS arm64 builds, explicitly add the necessary
  # linker flags until dependencies are well defined. See
  # https://github.com/bazelbuild/bazel/issues/19730.
  write_to_bazelrc "build:macos --linkopt=-Wl,-undefined,dynamic_lookup"
  write_to_bazelrc "build:macos --host_linkopt=-Wl,-undefined,dynamic_lookup"

  write_to_bazelrc "build --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  # Reduce noise during build.
  write_to_bazelrc "build --cxxopt=-Wno-deprecated-declarations --host_cxxopt=-Wno-deprecated-declarations"
  write_to_bazelrc "build --cxxopt=-Wno-parentheses --host_cxxopt=-Wno-parentheses"
  write_to_bazelrc "build --cxxopt=-Wno-sign-compare --host_cxxopt=-Wno-sign-compare"

  write_to_bazelrc "test --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
  write_to_bazelrc "test --action_env PYTHON_VERSION=${PYTHON_VERSION}"
  write_to_bazelrc "test --test_timeout=300"

  bazel clean
  bazel build ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}" --action_env MACOSX_DEPLOYMENT_TARGET='11.0'
  bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}"

  DEST="${OUTPUT_DIR}"'/all_dist'
  mkdir -p "${DEST}"

  printf '=== Destination directory: %s\n' "${DEST}"

  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"

  printf '%s : "=== Using tmpdir: %s\n' "$(date)" "${TMPDIR}"

  printf "=== Copy grain files\n"

  cp README.md "${TMPDIR}"
  cp setup.py "${TMPDIR}"
  cp pyproject.toml "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L --exclude="__pycache__/*" grain "${TMPDIR}"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/grain "${TMPDIR}"

  previous_wd="$(pwd)"
  cd "${TMPDIR}"
  printf '%s : "=== Building wheel\n' "$(date)"
  if [ "$(uname)" = "Darwin" ]; then
    "$PYTHON_BIN" setup.py bdist_wheel --python-tag py3"${PYTHON_MINOR_VERSION}" --plat-name macosx_11_0_"$(uname -m)"
  else
    "$PYTHON_BIN" setup.py bdist_wheel --python-tag py3"${PYTHON_MINOR_VERSION}"
  fi

  cp 'dist/'*.whl "${DEST}"

  if [ -n "${AUDITWHEEL_PLATFORM}" ]; then
    printf '%s : "=== Auditing wheel\n' "$(date)"
    auditwheel repair --plat "${AUDITWHEEL_PLATFORM}" -w dist dist/*.whl
  fi

  printf '%s : "=== Listing wheel\n' "$(date)"
  ls -lrt 'dist/'*.whl
  cp 'dist/'*.whl "${DEST}"
  cd "${previous_wd}"

  printf '%s : "=== Output wheel file is in: %s\n' "$(date)" "${DEST}"

  # Install grain from the wheel and run smoke tests.
  $PYTHON_BIN -m pip install --find-links=/tmp/grain/all_dist grain
  $PYTHON_BIN -m pip install tensorflow jax
  $PYTHON_BIN grain/_src/core/smoke_test_with_tf.py
  $PYTHON_BIN grain/_src/core/smoke_test_with_jax.py
}

main "$@"