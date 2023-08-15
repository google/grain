#!/bin/bash
# build wheel for python version specified in $PYTHON

set -e -x

export PYTHON_VERSION="${PYTHON_VERSION}"
export USE_BAZEL_VERSION="${BAZEL_VERSION}"
PYTHON="python${PYTHON_VERSION}"

function main() {
  bazel clean
  bazel build ... --action_env PYTHON_BIN_PATH="/usr/bin/$PYTHON"
  bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="/usr/bin/$PYTHON"

  DEST="/tmp/grain/all_dist"
  mkdir -p "${DEST}"

  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy grain files"

  cp pyproject.toml "${TMPDIR}"
  cp LICENSE "${TMPDIR}"
  rsync -avm -L --exclude="__pycache__/*" grain "${TMPDIR}"
  rsync -avm -L  --include="*.so" --include="*_pb2.py" \
    --exclude="*.runfiles" --exclude="*_obj" --include="*/" --exclude="*" \
    bazel-bin/grain "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  ${PYTHON} -m build
  cp dist/*.whl "${DEST}"
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
