#!/bin/bash

function build_and_test_grain() {
  SOURCE_DIR=$1
  OUTPUT_DIR=$2
  mkdir -p ${OUTPUT_DIR}

  # Automatically decide which platform to build for by checking on which
  # platform this runs.
  ARCH=$(uname -m)
  AUDITWHEEL_PLATFORM="manylinux2014_${ARCH}"

  # Using a previous version of Blaze to avoid:
  # https://github.com/bazelbuild/bazel/issues/8622
  export BAZEL_VERSION="5.4.0"

  # Copybara export to OUTPUT_DIR.
  copybara ${SOURCE_DIR}/third_party/py/grain/oss/copy.bara.sky local .. \
    --init-history --folder-dir=$OUTPUT_DIR --ignore-noop

  PYTHON_MAJOR_VERSION=3
  
  # TODO(iindyk): Build wheels for Python 3.12 once array_record supports it.
  for PYTHON_MINOR_VERSION in 9 10 11
  do
    PYTHON_VERSION=$PYTHON_MAJOR_VERSION$PYTHON_MINOR_VERSION
    docker rmi -f grain:${PYTHON_VERSION}
    docker rm -f grain
    DOCKER_BUILDKIT=1 docker build --progress=plain --no-cache \
      --build-arg ARCH=${ARCH} \
      --build-arg AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
      --build-arg BAZEL_VERSION=${BAZEL_VERSION} \
      -t grain:${PYTHON_VERSION} - < ${OUTPUT_DIR}/grain/oss/build.Dockerfile

    docker run --rm -a stdin -a stdout -a stderr \
      --env PYTHON_VERSION=$PYTHON_MAJOR_VERSION.$PYTHON_MINOR_VERSION \
      --env PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} \
      --env PYTHON_MINOR_VERSION=${PYTHON_MINOR_VERSION} \
      --env BAZEL_VERSION=${BAZEL_VERSION} \
      --env AUDITWHEEL_PLATFORM=${AUDITWHEEL_PLATFORM} \
      -v $OUTPUT_DIR:/tmp/grain \
      --name grain grain:${PYTHON_VERSION} \
      bash grain/oss/build_whl.sh
  done

  ls ${OUTPUT_DIR}/all_dist/*.whl
}