# Constructs the environment within which we will build the grain pip wheels.


ARG AUDITWHEEL_PLATFORM

FROM quay.io/pypa/${AUDITWHEEL_PLATFORM}
LABEL maintainer="Grain team <grain-dev@google.com>"

ARG PYTHON_VERSION
ARG BAZEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN yum install -y rsync

ENV PYTHON_BIN_PATH=/opt/python/cp${PYTHON_VERSION}-cp${PYTHON_VERSION}/bin
ENV PYTHON_BIN=${PYTHON_BIN_PATH}/python
ENV PATH="${PYTHON_BIN_PATH}:${PATH}"

# Download the correct bazel version and make sure it's on path.
RUN BAZEL_ARCH_SUFFIX="$(uname -m | sed s/aarch64/arm64/)" \
  && curl -sSL --fail -o /usr/local/bin/bazel "https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-linux-$BAZEL_ARCH_SUFFIX" \
  && chmod a+x /usr/local/bin/bazel

# If the context (Dockerfile) folder has array_record subfolder, then we expect
# it to have array_record wheels that we will install. Otherwise we install
# released array_record.
# We use glob pattern in the directory name to only copy the directory if it
# exists.
COPY /array_recor[d] /tmp/grain/array_record
RUN --mount=type=cache,target=/root/.cache \
  $PYTHON_BIN -m pip install -U \
    --find-links=/tmp/grain/array_record array_record;

# Install dependencies needed for grain.
RUN --mount=type=cache,target=/root/.cache \
  $PYTHON_BIN -m pip install -U \
    absl-py \
    build \
    cloudpickle \
    dm-tree \
    etils[epath,epy] \
    jaxtyping \
    more-itertools>=9.1.0 \
    numpy;

# Install dependencies needed for grain tests
RUN --mount=type=cache,target=/root/.cache \
  $PYTHON_BIN -m pip install -U \
    attrs \
    auditwheel \
    dill \
    jax \
    jaxlib \
    tensorflow \
    tensorflow-datasets;

WORKDIR "/tmp/grain"