#!/bin/bash
# Run the gcsfuse installer script
./install_gcsfuse.sh

# Check if the installer script was successful
if [ $? -ne 0 ]; then
  echo "gcsfuse installation failed. Exiting."
  exit 1
fi

/setup_gcsfuse.sh

pwd
ls . | grep load_from_gcsfuse

/load_from_gcsfuse --mount_path=${MOUNT_PATH}