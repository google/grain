#!/bin/bash
# build wheel for python version specified in $PYTHON_VERSION

set -e -x

CP_VERSION="cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}"
PYTHON_BIN_PATH="/opt/python/${CP_VERSION}-${CP_VERSION}/bin/python"

function main() {
  bazel clean
  bazel build ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN_PATH}"
  bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN_PATH}"

  DEST="/tmp/grain/all_dist"
  mkdir -p "${DEST}"

  echo "=== Destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy grain files"

  cp README.md "${TMPDIR}"
  cp setup.py "${TMPDIR}"
  cp pyproject.toml "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L --exclude="__pycache__/*" grain "${TMPDIR}"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/grain "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  "python${PYTHON_VERSION}" setup.py bdist_wheel --python-tag py3${PYTHON_MINOR_VERSION}
  cp dist/*.whl "${DEST}"

  echo $(date) : "=== Auditing wheel"
  auditwheel repair --plat ${AUDITWHEEL_PLATFORM} -w dist dist/*.whl

  echo $(date) : "=== Listing wheel"
  ls -lrt dist/*.whl
  cp dist/*.whl "${DEST}"
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
