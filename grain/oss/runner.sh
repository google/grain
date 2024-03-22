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


source ${SOURCE_DIR}/third_party/py/grain/oss/runner_common.sh
build_and_test_grain "${OUTPUT_DIR}"
