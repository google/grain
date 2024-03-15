#!/bin/bash

set -x -e

SOURCE_DIR=$PWD
OUTPUT_DIR="/tmp/grain"

source ${SOURCE_DIR}/third_party/py/grain/oss/runner_common.sh

build_and_test_grain "${SOURCE_DIR}" "${OUTPUT_DIR}"
