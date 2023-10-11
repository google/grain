# Constructs the environment within which we will build the grain pip wheels.
#
# From /tmp/grain,
# ❯ DOCKER_BUILDKIT=1 docker build \
#     --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
#     -t grain:${PYTHON_VERSION} - < grain/oss/build.Dockerfile
# ❯ docker run --rm -it -v /tmp/grain:/tmp/grain \
#      grain:${PYTHON_VERSION} bash

FROM ubuntu:22.04
LABEL maintainer="Grain team <grain-dev@google.com>"

# Declare args after FROM because the args declared before FROM can't be used in
# any instructions after a FROM
ARG PYTHON_VERSION
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends software-properties-common
RUN apt update && apt install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        rename \
        rsync \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup python
RUN apt-get update && apt-get install -y \
    python3-dev python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python${PYTHON_VERSION} -m pip install pip --upgrade && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 0

# Install bazel
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Install pip dependencies needed for grain
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

# Install pip dependencies needed for grain tests
RUN --mount=type=cache,target=/root/.cache \
  python${PYTHON_VERSION} -m pip install -U \
    dill \
    jax \
    jaxlib \
    tensorflow \
    tensorflow-datasets;

WORKDIR "/tmp/grain"