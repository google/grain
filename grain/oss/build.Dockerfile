# Constructs the environment within which we will build the grain pip wheels.
#
# From /tmp/grain,
# ❯ DOCKER_BUILDKIT=1 docker build \
#     --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
#     -t grain:${PYTHON_VERSION} - < grain/oss/build.Dockerfile
# ❯ docker run --rm -it -v /tmp/grain:/tmp/grain \
#      grain:${PYTHON_VERSION} bash

FROM quay.io/pypa/manylinux2014_x86_64
LABEL maintainer="Grain team <grain-dev@google.com>"

ARG PYTHON_MAJOR_VERSION
ARG PYTHON_MINOR_VERSION
ARG PYTHON_VERSION
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN yum install -y rsync

ENV PATH="/opt/python/cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}/bin:${PATH}"

# Install bazel
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Install dependencies needed for grain
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U \
    absl-py \
    array_record \
    build \
    cloudpickle \
    dm-tree \
    etils[epath] \
    jaxtyping \
    more-itertools>=9.1.0 \
    numpy;

# Install dependencies needed for grain tests
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U \
    auditwheel \
    dill \
    jax \
    jaxlib \
    tensorflow \
    tensorflow-datasets;

WORKDIR "/tmp/grain"