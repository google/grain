#!/bin/bash

# Script to check if gcsfuse is installed and run an installer if not.

# Check if gcsfuse is already installed and in PATH
if command -v gcsfuse >/dev/null 2>&1; then
    echo "gcsfuse is already installed."
    echo "Installed version:"
    gcsfuse --version # Display version for confirmation
else
    echo "gcsfuse is not found in PATH."
    apt-get install -y coreutils
    echo $0
    echo "Attempting to install gcsfuse..."
    apt-get update && apt-get install -y sudo
    (sudo bash || bash) <<'EOF'
    apt update && \
    apt install -y numactl lsb-release gnupg curl net-tools iproute2 procps lsof git ethtool && \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt update -y && apt -y install gcsfuse
    rm -rf /var/lib/apt/lists/*
EOF

fi

exit 0 # Exit successfully

