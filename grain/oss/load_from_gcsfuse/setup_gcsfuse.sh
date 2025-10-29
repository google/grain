#!/bin/bash
# Description:
# bash setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=/tmp/gcsfuse FILE_PATH=/tmp/gcsfuse/my_dataset

set -e

export DATASET_GCS_BUCKET=grain-benchmark-example-bucket MOUNT_PATH=/test/mmmu

echo "Setting up gcsfuse for $DATASET_GCS_BUCKET"
echo "Mounting to $MOUNT_PATH"

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ -z ${DATASET_GCS_BUCKET} || -z ${MOUNT_PATH} ]]; then
  echo "Please set arguments: DATASET_GCS_BUCKET and MOUNT_PATH"
  exit 1
fi

if [[ "$DATASET_GCS_BUCKET" =~ gs:\/\/ ]] ; then
    DATASET_GCS_BUCKET="${DATASET_GCS_BUCKET/gs:\/\//}"
    echo "Removed gs:// from GCS bucket name, GCS bucket is $DATASET_GCS_BUCKET"
fi

if [[ -d ${MOUNT_PATH} ]]; then
  echo "$MOUNT_PATH exists, removing..."
  fusermount -u $MOUNT_PATH || rm -rf $MOUNT_PATH
fi

mkdir -p $MOUNT_PATH

# see https://cloud.google.com/storage/docs/gcsfuse-cli for all configurable options of gcsfuse CLI
TIMESTAMP=$(date +%Y%m%d-%H%M)
gcsfuse -o ro --implicit-dirs --log-severity=debug \
        --type-cache-max-size-mb=-1 --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 \
        --log-file=$HOME/gcsfuse_$TIMESTAMP.json "$DATASET_GCS_BUCKET" "$MOUNT_PATH"

# Use ls to prefill the metadata cache: https://cloud.google.com/storage/docs/cloud-storage-fuse/performance#improve-first-time-reads
if [[ ! -z ${FILE_PATH} ]] ; then 
  FILE_COUNT=$(ls -R $FILE_PATH | wc -l)
  echo $FILE_COUNT files found in $FILE_PATH
fi

ls -R $MOUNT_PATH