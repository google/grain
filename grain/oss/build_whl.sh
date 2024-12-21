#!/bin/bash
# build wheel for python version specified in $PYTHON_VERSION

set -e -x

function main() {
  bazel clean
  bazel build ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}" --action_env MACOSX_DEPLOYMENT_TARGET=11.0
  bazel test --verbose_failures --test_output=errors ... --action_env PYTHON_BIN_PATH="${PYTHON_BIN}"

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
  plat_name=""
  if [[ "$(uname)" == "Darwin" ]]; then
    plat_name="--plat-name macosx_11_0_$(uname -m)"
  fi

  $PYTHON_BIN setup.py bdist_wheel --python-tag py3${PYTHON_MINOR_VERSION} $plat_name
  cp dist/*.whl "${DEST}"

  if [ -n "${AUDITWHEEL_PLATFORM}" ]; then
    echo $(date) : "=== Auditing wheel"
    auditwheel repair --plat ${AUDITWHEEL_PLATFORM} -w dist dist/*.whl
  fi

  echo $(date) : "=== Listing wheel"
  ls -lrt dist/*.whl
  cp dist/*.whl "${DEST}"
  popd

  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
