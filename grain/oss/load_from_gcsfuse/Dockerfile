FROM ubuntu:latest as builder
ENV PATH=/usr/local/bin:$PATH
# Set DEBIAN_FRONTEND to noninteractive to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    fuse \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Bazel
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg \
    && mv bazel.gpg /etc/apt/trusted.gpg.d/ \
    && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get update && apt-get install -y bazel

# Install gcsfuse
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y gcsfuse

# Add Bazel to PATH
ENV PATH="/root/.bazel/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy the entire workspace
COPY . .

# Build the load_from_gcsfuse binary
RUN bazel build //grain/oss/load_from_gcsfuse:load_from_gcsfuse

# Create mount point for GCS
RUN mkdir -p /mnt/gcs

# Set up proper permissions for FUSE
RUN echo 'user_allow_other' >> /etc/fuse.conf

# Make the setup script executable
RUN chmod +x grain/oss/test/setup_gcsfuse.sh

# Create a startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Run the gcsfuse setup script\n\
echo "Running gcsfuse setup..."\n\
./grain/oss/load_from_gcsfuse/setup_gcsfuse.sh\n\
\n\
# Run the Python binary\n\
echo "Starting load_from_gcsfuse..."\n\
exec ./bazel-bin/grain/oss/load_from_gcsfuse/load_from_gcsfuse "$@"' > /workspace/start.sh \
    && chmod +x /workspace/start.sh

# Set the entrypoint
ENTRYPOINT ["/workspace/start.sh"]

# Default command
CMD []
