# Constructs the environment within which we will build the grain pip wheels.
#
# From third_party/py/grain,
# ❯ DOCKER_BUILDKIT=1 docker build -t grain:latest - < oss/build.Dockerfile
# ❯ docker run --rm -it -v $HOME/tmp/grain:/tmp/grain grain:latest bash    

ARG base_image="tensorflow/build:2.11-python3.9"
FROM $base_image
LABEL maintainer="Grain team <Grain@google.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install supplementary Python interpreters
RUN mkdir /tmp/python
RUN --mount=type=cache,target=/var/cache/apt \
  apt update && \
  apt install -yqq \
    apt-utils \
    build-essential \
    checkinstall \
    libffi-dev \
    neovim

ARG bazel_version=5.1.1
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
# 3.9 is the built-in interpreter version in this image.
# RUN for v in 3.8.15; do \
#     wget "https://www.python.org/ftp/python/$v/Python-${v}.tar.xz" && \
#     rm -rf "/tmp/python${v}" && mkdir -p "/tmp/python${v}" && \
#     tar xvf "Python-${v}.tar.xz" -C "/tmp/python${v}" && \
#     cd "/tmp/python${v}/Python-${v}" && \
#     ./configure 2>&1 >/dev/null && \
#     make -j8 altinstall 2>&1 >/dev/null && \
#     ln -sf "/usr/local/bin/python${v%.*}" "/usr/bin/python${v%.*}"; \
#   done

# For each python interpreter, install pip dependencies needed for array_record
RUN --mount=type=cache,target=/root/.cache \
#  for p in 3.8 3.9; do \
for p in 3.9; do \
    python${p} -m pip install -U pip && \
    python${p} -m pip install -U \
      absl-py \
      array_record \
      auditwheel \
      etils \
      google-re2 \
      jax \
      jaxlib \
      numpy \
      orbax \
      patchelf \
      seqio \
      setuptools \
      twine \
      typing_extensions \
      wheel; \
  done

WORKDIR "/tmp/grain"
