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

  write_to_bazelrc "build --@rules_python//python/config_settings:python_version=${PYTHON_VERSION}"
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
}

main "$@"