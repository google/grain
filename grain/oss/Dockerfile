# Constructs the environment within which we will build the grain pip wheels.


ARG AUDITWHEEL_PLATFORM

FROM quay.io/pypa/${AUDITWHEEL_PLATFORM}
LABEL maintainer="Grain team <grain-dev@google.com>"

ARG ARCH
ARG PYTHON_VERSION
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN yum install -y rsync

ENV PYTHON_BIN=/opt/python/cp${PYTHON_VERSION}-cp${PYTHON_VERSION}/bin
ENV PATH="${PYTHON_BIN}:${PATH}"

# Install bazel
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux ${ARCH}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux ${ARCH}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-$ARCH.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-$ARCH.sh

# If the context (Dockerfile) folder has array_record subfolder, then we expect
# it to have array_record wheels that we will install. Otherwise we install
# released array_record.
# We use glob pattern in the directory name to only copy the directory if it
# exists.
COPY /array_recor[d] /tmp/grain/array_record
RUN --mount=type=cache,target=/root/.cache \
  ${PYTHON_BIN}/python -m pip install -U \
    --find-links=/tmp/grain/array_record array_record;

# Install dependencies needed for grain.
RUN --mount=type=cache,target=/root/.cache \
  ${PYTHON_BIN}/python -m pip install -U \
    absl-py \
    build \
    cloudpickle \
    dm-tree \
    etils[epath] \
    jaxtyping \
    more-itertools>=9.1.0 \
    numpy;

# Install dependencies needed for grain tests
RUN --mount=type=cache,target=/root/.cache \
  ${PYTHON_BIN}/python -m pip install -U \
    auditwheel \
    dill \
    jax \
    jaxlib \
    tensorflow \
    tensorflow-datasets;

WORKDIR "/tmp/grain"