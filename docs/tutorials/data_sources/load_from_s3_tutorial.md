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

<!-- #region id="Xz3HnUBqWlWf" -->
## Reading from AWS S3

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/dataset_load_from_s3_tutorial.ipynb)

This document outlines how to read data from an Amazon S3 bucket and construct a Grain pipeline. We will leverage [S3 Mountpoint](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mountpoint.html), a service provided by AWS. S3 Mountpoint enables you to mount your S3 bucket as a local file system, allowing you to access and read data as if it were stored locally.
<!-- #endregion -->

<!-- #region id="8Q4NLlCnWlWf" -->
### Install Mountpoint for Amazon S3
<!-- #endregion -->

```python id="K6UTOyamWlWf"
!wget https://s3.amazonaws.com/mountpoint-s3-release/latest/x86_64/mount-s3.deb
```

```python id="iHA-C85NhwFJ"
!sudo apt-get install -y ./mount-s3.deb
```

<!-- #region id="Y4cIxXULe8kB" -->
### Configure AWS credentials
<!-- #endregion -->

```python id="8fhEOwxcWlWf"
!pip install aws configure
!pip install awscli
```

```python id="5Lt_644G7G9R"
!aws configure
```

<!-- #region id="qRezs5v-e8kB" -->
### Mount your S3 bucket to your local filepath
<!-- #endregion -->

```python id="G6boYrD5WlWf"
!mount-s3 <your-s3-bucket> /path/to/mount/files
```

<!-- #region id="KoquHCPMe8kB" -->
### Install Grain and other dependencies
<!-- #endregion -->

```python id="3BZP9fBiWlWf"
!pip install grain
!pip install array_record
```

<!-- #region id="4eESJ_qFic0B" -->
### Write temp ArrayRecord files to the bucket
<!-- #endregion -->

```python id="xVGVuDKNic0B"
from array_record.python import array_record_module

digits = [b"1", b"2", b"3", b"4", b"5"]

writer = array_record_module.ArrayRecordWriter("/path/to/mount/files/data.array_record")
for i in digits:
  writer.write(i)
writer.close()
```

<!-- #region id="KtAwV_Sgic0B" -->
### Read ArrayRecord files using Grain
<!-- #endregion -->

```python id="3l4Pnc4bWlWf"
import grain
from pprint import pprint

source =  grain.sources.ArrayRecordDataSource(paths="/path/to/mount/files/data.array_record")

dataset = (
    grain.MapDataset.source(source)
    .shuffle(seed=10)  # Shuffles globally.
    .batch(batch_size=2)  # Batches consecutive elements.
)

pprint(list(dataset))
```
