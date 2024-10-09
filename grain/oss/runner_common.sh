#!/bin/bash

# Builds Grain from source code located in SOURCE_DIR producing wheels under
# $SOURCE_DIR/all_dist.
function build_and_test_grain() {
  SOURCE_DIR=$1

  # Automatically decide which platform to build for by checking on which
  # platform this runs.
  AUDITWHEEL_PLATFORM="manylinux2014_$(uname -m)"

  # Using a previous version of Blaze to avoid:
  # https://github.com/bazelbuild/bazel/issues/8622
  export BAZEL_VERSION="5.4.0"

  # Build wheels for multiple Python minor versions.
  PYTHON_MAJOR_VERSION=3
  for PYTHON_MINOR_VERSION in 10 11 12
  do
    PYTHON_VERSION=$PYTHON_MAJOR_VERSION$PYTHON_MINOR_VERSION
    docker rmi -f grain:${PYTHON_VERSION}
    docker rm -f grain
    DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
      --build-arg AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
      --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
      -t grain:${PYTHON_VERSION} ${SOURCE_DIR}/grain/oss

    docker run --rm -a stdin -a stdout -a stderr \
      --env PYTHON_VERSION=$PYTHON_MAJOR_VERSION.$PYTHON_MINOR_VERSION \
      --env PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
      --env PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
      --env BAZEL_VERSION=${BAZEL_VERSION} \
      --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      -v $SOURCE_DIR:/tmp/grain \
      --name grain grain:${PYTHON_VERSION} \
      bash grain/oss/build_whl.sh
  done

  ls ${SOURCE_DIR}/all_dist/*.whl
}