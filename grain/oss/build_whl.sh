#!/bin/bash
# build wheel for python version specified in $PYTHON

set -e -x

export PYTHON_MINOR_VERSION="${PYTHON_MINOR_VERSION}"
PYTHON="python3${PYTHON_MINOR_VERSION:+.$PYTHON_MINOR_VERSION}"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function main() {
  # Remove .bazelrc if it already exists
  [ -e .bazelrc ] && rm .bazelrc

  write_to_bazelrc "build -c opt"
  write_to_bazelrc "build --cxxopt=-std=c++17"
  write_to_bazelrc "build --host_cxxopt=-std=c++17"
  write_to_bazelrc "build --linkopt=\"-lrt -lm\""
  write_to_bazelrc "build --experimental_repo_remote_exec"
  write_to_bazelrc "build --action_env=PYTHON_BIN_PATH=\"/usr/bin/$PYTHON\""
  write_to_bazelrc "build --action_env=PYTHON_LIB_PATH=\"/usr/lib/$PYTHON\""
  write_to_bazelrc "build --python_path=\"/usr/bin/$PYTHON\""

  TF_NEED_CUDA=0
  echo 'Using installed tensorflow'
  TF_CFLAGS=( $(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
  TF_LFLAGS="$(${PYTHON} -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

  write_action_env_to_blazerc "TF_HEADER_DIR" ${TF_CFLAGS:2}
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
  SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
  if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
    if [[ "$(uname)" == "Darwin" ]]; then
      SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
    else
      SHARED_LIBRARY_NAME="libtensorflow_framework.so"
    fi
  fi
  write_action_env_to_blazerc "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR}
  write_action_env_to_blazerc "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME}
  write_action_env_to_blazerc "TF_NEED_CUDA" ${TF_NEED_CUDA}
}