#!/bin/bash

set -x -e

# Expects the source code to be under $PWD/third_party/py/grain. The wheels are
# going to be under $OUTPUT_DIR/all_dist.
SOURCE_DIR=$PWD
OUTPUT_DIR="/tmp/grain"
mkdir -p ${OUTPUT_DIR}

# Copybara export to OUTPUT_DIR.
copybara ${SOURCE_DIR}/third_party/py/grain/oss/copy.bara.sky local .. \
  --init-history --folder-dir=$OUTPUT_DIR --ignore-noop

# If BUILD_ARRAY_RECORD is set, download array_record source from github and
# build it's wheels. In such case, the wheels will also be installed as Grain's
# dependency.
if [ -n "${BUILD_ARRAY_RECORD}" ]; then
  export ARRAY_RECORD_OUTPUT_DIR="/tmp/array_record"
  git clone https://github.com/google/array_record ${ARRAY_RECORD_OUTPUT_DIR}
  pushd ${ARRAY_RECORD_OUTPUT_DIR}
  # Use ArrayRecord commit when build was still working.
  git reset --hard ef9be1b9de19e9e9ca5c272490a2fca4afb3c4ec
  git apply ${OUTPUT_DIR}/grain/oss/array_record/WORKSPACE.patch
  git apply ${OUTPUT_DIR}/grain/oss/array_record/build_whl.patch
  git apply ${OUTPUT_DIR}/grain/oss/array_record/runner_common.patch
  git apply ${OUTPUT_DIR}/grain/oss/array_record/Dockerfile.patch
  git apply ${OUTPUT_DIR}/grain/oss/array_record/setup.patch
  git apply ${OUTPUT_DIR}/grain/oss/array_record/array_record_reader.patch

  source ${ARRAY_RECORD_OUTPUT_DIR}/oss/runner_common.sh
  PLATFORM="$(uname)"
  if [[ "$PLATFORM" == "Darwin" ]]; then
    build_and_test_array_record_macos "${ARRAY_RECORD_OUTPUT_DIR}"
  else
    build_and_test_array_record_linux "${ARRAY_RECORD_OUTPUT_DIR}"
  fi

  # array-record scripts override these variables, so we need to reset them.
  popd
  SOURCE_DIR=$PWD
  OUTPUT_DIR="/tmp/grain"
  mkdir -p ${OUTPUT_DIR}/grain/oss/array_record
  cp -r ${ARRAY_RECORD_OUTPUT_DIR}/all_dist/* ${OUTPUT_DIR}/grain/oss/array_record
fi


source ${SOURCE_DIR}/third_party/py/grain/oss/runner_common.sh
PLATFORM="$(uname)"
if [[ "$PLATFORM" == "Darwin" ]]; then
  build_and_test_grain_macos "${OUTPUT_DIR}"
else
  build_and_test_grain_linux "${OUTPUT_DIR}"
fi
