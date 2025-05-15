---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- disableFinding(LINE_OVER_80) -->
<!-- #region id="HV4SMb5j_Y22" -->
# Reading from GCS

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/data_sources/load_from_gcs_tutorial.ipynb)

This document demonstrates how to access and load data from Google Cloud Storage using Grain. To achieve this, we'll utilize Cloud Storage [FUSE](https://cloud.google.com/storage/docs/cloud-storage-fuse/overview), an adapter that allows you to mount GCS buckets as local file systems. By using Cloud Storage FUSE to mount GCS buckets as local file systems, you can access cloud storage data just like local files.
<!-- #endregion -->

<!-- #region id="IDFo-1Dx_Y23" -->
## Mount a Cloud Storage location into the local filesystem
<!-- #endregion -->

```python id="h6HqcZSQ_Y23"
# Authenticate.
from google.colab import auth
auth.authenticate_user()
```



<!-- #region id="0Ajla2LNdRVq" -->
The gcsfuse CLI offers various configurable options, detailed at https://cloud.google.com/storage/docs/gcsfuse-cli. Utilizing certain options, such as the caching features described at https://cloud.google.com/storage/docs/cloud-storage-fuse/caching, can enhance read performance and lower costs. For instance, MaxText setup gcsfuse flags ([MaxText gcsfuse setting link](https://github.com/AI-Hypercomputer/maxtext/blob/4e36b61cf40698224c5251c4aa4086df24140bd1/setup_gcsfuse.sh#L48)) to reduce data loading time for training. We advise users to consider adopting similar settings or customizing their own gcsfuse options.
<!-- #endregion -->

```python id="bqz6cD7xl7F3"
# Mount a Cloud Storage bucket or location, without the gs:// prefix.
mount_path = "my-bucket"  # or a location like "my-bucket/path/to/mount"
local_path = f"./mnt/gs/{mount_path}"

!mkdir -p {local_path}
# The flags below are configured to improve GCS data loading performance. Users are encouraged to explore alternative settings and we would greatly appreciate any feedback or insights shared with the Grain team.
!gcsfuse --implicit-dirs --type-cache-max-size-mb=-1 --stat-cache-max-size-mb=-1 --kernel-list-cache-ttl-secs=-1 --metadata-cache-ttl-secs=-1 {mount_path} {local_path}
```

```python id="j2e8nv0j_Y23"
# Then you can access it like a local path.
!ls -lh {local_path}
```

<!-- #region id="BpCK0dm4_Y23" -->
## Read files using Grain

If your data is in an ArrayRecord file, you can directly load it using `grain.sources.ArrayRecordDataSource`. For information on handling other file formats, please see the Grain data sources documentation at: https://google-grain.readthedocs.io/en/latest/data_sources.html
<!-- #endregion -->

```python id="yisjIpbZ_Y23"
# Install Grain.
!pip install grain
```

```python id="pvNTx6sL_Y23"
import grain

source = grain.sources.ArrayRecordDataSource(local_path+"/local_file_name")

# Create a dataset from the data source then process the data.
dataset = (
    grain.MapDataset.source(source)
    .shuffle(seed=10)  # Shuffles globally.
    .batch(batch_size=2)  # Batches consecutive elements.
)
```

```python id="bJIYx60H_Y23"
# Output a record at a random index
print(dataset[10])
```
