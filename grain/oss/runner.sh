#!/bin/sh

set -x -e

# Expects the source code to be under $PWD/third_party/py/grain. The wheels are
# going to be under $OUTPUT_DIR/all_dist.
SOURCE_DIR="${PWD}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/grain}"
ORIGINAL_OUTPUT_DIR="${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Copybara export to OUTPUT_DIR.
copybara "${SOURCE_DIR}"'/third_party/py/grain/oss/copy.bara.sky' local .. \
  --init-history --folder-dir="$OUTPUT_DIR" --ignore-noop

# If BUILD_ARRAY_RECORD is set, download array_record source from github and
# build it's wheels. In such case, the wheels will also be installed as Grain's
# dependency.
if [ -n "${BUILD_ARRAY_RECORD}" ]; then
  export ARRAY_RECORD_OUTPUT_DIR="/tmp/array_record"
  git clone https://github.com/google/array_record "${ARRAY_RECORD_OUTPUT_DIR}"
  previous_wd="$(pwd)"
  cd "${ARRAY_RECORD_OUTPUT_DIR}"
  # Broke on a previous commit, so taking care to pin to a working one:
  git reset --hard 7e299eae0db0d7bfc20f7c1e1548bf86cdbfef5e

  . "${ARRAY_RECORD_OUTPUT_DIR}"'/oss/runner_common.sh'
  PLATFORM="$(uname)"
  if [ "$PLATFORM" = "Darwin" ]; then
    build_and_test_array_record_macos "${ARRAY_RECORD_OUTPUT_DIR}"
  else
    build_and_test_array_record_linux "${ARRAY_RECORD_OUTPUT_DIR}"
  fi

  # array-record scripts override these variables, so we need to reset them.
  cd "${previous_wd}"
  SOURCE_DIR="${PWD}"
  OUTPUT_DIR="${ORIGINAL_OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"'/grain/oss/array_record'
  cp -r "${ARRAY_RECORD_OUTPUT_DIR}"'/all_dist/'* "${OUTPUT_DIR}"'/grain/oss/array_record'
fi


. "${SOURCE_DIR}"'/third_party/py/grain/oss/runner_common.sh'
PLATFORM="$(uname)"
if [ "$PLATFORM" = "Darwin" ]; then
  build_and_test_grain_macos "${OUTPUT_DIR}"
else
  build_and_test_grain_linux "${OUTPUT_DIR}"
fi
