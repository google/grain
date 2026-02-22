"""Load data from GCSfuse."""

import logging
import os
import subprocess
import time
from typing import Any, Iterable
from absl import app
from absl import flags
import grain

_DATASET_GCS_BUCKET = flags.DEFINE_string(
    "gcs_bucket",
    default=None,
    required=True,
    help=(
        "Cloud Storage bucket or location, without the gs:// prefix, for"
        " example: my-bucket/my-folder"
    ),
)

_MOUNT_PATH = flags.DEFINE_string(
    "mount_path",
    default=None,
    required=True,
    help="The local path to mount the Cloud Storage bucket to.",
)

_DEFAULT_BATCH_SIZE = 256


def _get_grain_datasource(file_path):
  """get_grain_datasource returns a grain.DataSource."""
  return grain.sources.ArrayRecordDataSource(file_path)


def _get_data_iterable(file_path) -> Iterable[Any]:
  """get_data_iterable returns an Iterable over _dataset from _datasource."""

  try:
    # Create the pygrain dataset.
    dataset = (
        grain.MapDataset.source(_get_grain_datasource(file_path=file_path))
        .shuffle(seed=42)
        .batch(batch_size=_DEFAULT_BATCH_SIZE)
        .repeat()
        .to_iter_dataset()
    )
    return dataset
  except Exception:  # pylint: disable=broad-exception-caught
    logging.exception("Failed to create dataset: %s")
    return iter([])


def _mount_gcs_bucket(gcs_bucket: str, mount_path: str) -> None:
  """Mounts a Cloud Storage bucket to a local path."""
  gcs_command = (
      "gcsfuse --implicit-dirs --implicit-dirs --type-cache-max-size-mb=-1"
      " --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1"
      f" --metadata-cache-ttl-secs=-1 {mount_path} {gcs_bucket}"
  )
  print(f"gcs_command: {gcs_command}")
  result = subprocess.run(
      gcs_command, shell=True, capture_output=True, text=True, check=False
  )

  # Check if the command was successful
  print("*" * 80)
  print(result.returncode)
  if result.returncode == 0:
    print("GCS Command executed successfully!")
    print("Output:\n", result.stdout)
  else:
    print("Command failed!")
    print("Error:\n", result.stderr)

  result = subprocess.run(
      f"ls -l {mount_path}",
      shell=True,
      capture_output=True,
      text=True,
      check=False,
  )
  print("*" * 80)
  print(result.stdout)


def main(_) -> None:
  if _DATASET_GCS_BUCKET.value is None:
    raise ValueError("The dataset GCS bucket is required.")
  if _MOUNT_PATH.value is None:
    raise ValueError("The mount path is required.")
  gcs_bucket = _DATASET_GCS_BUCKET.value
  mount_path = _MOUNT_PATH.value
  if gcs_bucket.startswith("gs://"):
    gcs_bucket = gcs_bucket.removeprefix("gs://")

  print(f"gcs_bucket: {gcs_bucket}")
  print(f"mount_path: {mount_path}")

  if not os.path.exists(mount_path):
    os.makedirs(mount_path)
    print(f"Created directory: {mount_path}")
  else:
    print(f"Directory already exists: {mount_path}")

  # _mount_gcs_bucket(gcs_bucket, mount_path)
  file_paths = [f for f in os.listdir(mount_path) if ".array_record" in f]
  print(f"file_paths: {file_paths}")
  start = time.time()
  # data_iter = iter(_get_data_iterable(file_paths))
  create_dataset_time = time.time() - start
  print(f"create_dataset_time: {create_dataset_time}")
  # start = time.time()
  # next_n = next(data_iter)
  # first_batch_time = time.time() - start
  start = time.time()
  curr = start
  examples = 0
  while curr < start + 60:
    try:
      # n = next(data_iter)
      examples += _DEFAULT_BATCH_SIZE
      curr = time.time()
    except StopIteration:
      logging.warning("Dataset iterator exhausted before duration ended.")
      break
  examples_per_sec = examples / (curr - start)
  print(f"examples_per_sec: {examples_per_sec}")


if __name__ == "__main__":
  app.run(main)
