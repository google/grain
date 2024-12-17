---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "qGiXX-sg4l9o"}

# `DataLoader` guide

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/data_loader_tutorial.ipynb)

+++ {"id": "rKvQbK6LK7Yd"}

## Install Grain and tutorial dependencies

```{code-cell}
:id: EZ9EXOZKehes

!pip install grain tfds-nightly opencv-python matplotlib orbax-checkpoint

import cv2
import orbax.checkpoint as ocp
import grain.python as grain
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ.pop('TFDS_DATA_DIR', None)
import tensorflow_datasets as tfds
```

+++ {"id": "aV6eMfzXe-Qz"}

## Concepts
The `DataLoader` class is responsible for reading and transforming input data records. Users need to iterate through the `DataLoader` to get output elements.

The `DataLoader` uses a **Sampler** to determine which records to read next, a **DataSource** to read the records, and applies **Transformations** to the read records to produce output elements. We present these concepts in the following sections.

+++ {"id": "KEYc7qb1gVPj"}

## `Sampler`
A `Sampler` is responsible for determining which records to read next. This involves applying global transformations such as shuffling records, repeating for multiple epochs and sharding across multiple machines. The sampler is an `Iterator` which produces metadata objects containing the following information:
* `index`: a monotonically increasing number, unique for each record. It keeps track of the pipeline process and will be used for checkpointing.
* `record_key`: a reference to the record in the serialized file format.
* `rng`: a per-record Random Number Generator to apply random transformations.

Grain provides an `IndexSampler` implementation, which we explore below.

```{code-cell}
---
executionInfo:
  elapsed: 59
  status: ok
  timestamp: 1734464259034
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: qk4MtXvXe1Bl
outputId: e851d69c-0bfa-4af1-e4ad-bb14d07cd4a3
---
# Setting `num_records` to 5 to limit results.
# For full dataset, set to `len(data_source)`.
index_sampler_example = grain.IndexSampler(
    num_records=5,
    num_epochs=2,
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=True,
    seed=0)

# Iterator is consumed.
for record_metadata in index_sampler_example:
  print(record_metadata)

# Create new iterator.
index_sampler = grain.IndexSampler(
    num_records=5,
    num_epochs=2,
    shard_options=grain.ShardOptions(
        shard_index=0, shard_count=1, drop_remainder=True),
    shuffle=True,
    seed=0)
```

+++ {"id": "P9j4heLCioo8"}

## Data source
A data source is responsible for reading indvidual records from underlying files / storage system. We provide the following data sources:

*   `ArrayRecordDataSource`: reads records from [ArrayRecord](go/array-record-design) files.
*   `tfds.data_source`: data source for [TFDS](https://www.tensorflow.org/datasets) datasets without a TensorFlow dependency.


Below, we show an example using a TFDS data source, but using other data sources should be similar.

+++ {"id": "0AbECqVlLD_q"}

## TFDS Data source

```{code-cell}
---
executionInfo:
  elapsed: 38785
  status: ok
  timestamp: 1734464297879
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: y6Yav1dCg8Ny
outputId: 76f2a8f1-90b2-45ed-e2ba-97fce09b2a0f
---
# We use a small version of ImageNet data to spare disk and network
# usage for the demo. Prefer "imagenet2012"'s "train" split for a
# complete dataset.
source = tfds.data_source("imagenet_a", split="test")

print(f"# records: {len(source)}")

print(f"First record:\n{source[0]}")
```

+++ {"id": "YH5O7BWl8UGz"}

## Transformations

Grain `Transformations` represent transformations applied to input elements. We offer ready-to-use: `BatchTransform` and the following transformations in form of abstract classes:
* `MapTransform`
* `RandomMapTransform`
* `FilterTransform`

You'd need to create your concrete transformation by inheriting above classes.

**Note:**

1. Since Grain uses Python multiprocessing to parallelize work, transformation must be picklable, so they can be sent from the main process to the workers.
2. When using `BatchTransform` the `batch_size` is the global batch size if it is done before sharding, if it is done after sharding it is the batch size *per host* (this is typically the case when using `IndexSampler` with `DataLoader`).

```{code-cell}
:id: lRKLPtMK8w1A

class ResizeAndCrop(grain.MapTransform):

  def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image = element["image"]
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    image = image[:224, :224]
    element["image"] = image
    return element

transformations = [ResizeAndCrop()]
```

+++ {"id": "tj2tVIqc9MwM"}

## `DataLoader`

`DataLoader` is the glue between `Sampler`, `DataSource` and transformations. In addition, it is responsible for launching children processes to parallelize the processing of the input pipeline, collecting output elements from these processes and gracefully shutting them down at exit. Users need to iterate through the `DataLoader` to get processed elements (typically the output batches).

For quick experimentation, use `worker_count=0`, to run everything in a single process, saving the time to setup workers. When going to real training / evaluation, increase the number of workers to parallelize processing.

```{code-cell}
:id: RhcG6Ehs9uVy

data_loader = grain.DataLoader(
    data_source=source,
    operations=transformations,
    sampler=index_sampler,
    worker_count=0)
```

```{code-cell}
---
colab:
  height: 1000
executionInfo:
  elapsed: 2051
  status: ok
  timestamp: 1734464300160
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: aXN-iffE94Qc
outputId: 4174b044-e9f8-49d7-b83a-2e524f2bfae4
---
for element in data_loader:
  fig = plt.figure
  plt.imshow(element["image"])
  plt.show()
```

+++ {"id": "eODdpQcZeOfo"}

## Checkpointing
We provide `GrainCheckpointHandler` to checkpoint the iterator returned by Grain. It is recommended to use it with [Orbax](https://orbax.readthedocs.io), which can checkpoint both input pipeline and model and handles the edge cases for distributed training.

+++ {"id": "0JI0hvnUetWB"}

### Integration with Orbax
Orbax contains a powerful library for checkpointing various objects - incl. Flax models and Grain `DatasetIterators`. This makes it easy to checkpoint the `DatasetIterator` together with the model in a multihost environment. Orbax will take care of synchronizing all JAX processes.

```{code-cell}
---
executionInfo:
  elapsed: 496
  status: ok
  timestamp: 1734464300715
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: irJix4sJkNcf
outputId: 648ffc5d-088e-4747-da1d-f0bfd87e4360
---
data_iter = iter(data_loader)

num_steps = 5

# Read some elements.
for i in range(num_steps):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

```{code-cell}
---
executionInfo:
  elapsed: 74
  status: ok
  timestamp: 1734464309851
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: qRoTuxBNl8DB
outputId: e0868eb6-3412-4508-c665-074cd534a65d
---
mngr = ocp.CheckpointManager("/tmp/orbax")

!rm -rf /tmp/orbax

# Save the checkpoint
assert mngr.save(
    step=num_steps, args=grain.GrainCheckpointSave(data_iter), force=True)
# Checkpoint saving in Orbax is asynchronous by default, so we'll wait until
# finished before examining checkpoint.
mngr.wait_until_finished()

!ls -R /tmp/orbax
```

```{code-cell}
---
executionInfo:
  elapsed: 545
  status: ok
  timestamp: 1734464314411
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: JVt1F3JbkWrz
outputId: fa44fd08-d98e-43e2-9efe-37766db514e2
---
!cat /tmp/orbax/*/*/*.json
```

+++ {"id": "5TFV1ScvsuTw"}

Note: the checkpoint contains the string representation of the sampler and the data source. Checkpoints are only valid when loaded with the same sampler/data source and Grain uses the string representation for a basic check.

```{code-cell}
---
executionInfo:
  elapsed: 23
  status: ok
  timestamp: 1734464315897
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: wT04O9cdlpPJ
outputId: a1d350bc-e893-4535-895f-650f5cbd9c88
---
# Read more elements and advance the iterator
for i in range(5, 10):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

```{code-cell}
---
executionInfo:
  elapsed: 21
  status: ok
  timestamp: 1734464317504
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: Js3hheiGnykN
outputId: 31cc1d8d-9a89-4fa8-af7a-5c74650f118e
---
# Restore iterator from previously saved checkpoint
mngr.restore(num_steps, args=grain.GrainCheckpointRestore(data_iter))
```

```{code-cell}
---
executionInfo:
  elapsed: 340
  status: ok
  timestamp: 1734464320007
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: X6NT-drgoaJk
outputId: 591c4799-a1cb-418e-a08f-2f78e54e5282
---
# Iterator should be set back to start from 5.
for i in range(5, 10):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

+++ {"id": "btSRh4EL_Zbo"}

## Extras

+++ {"id": "LyIXwNWct7mQ"}

### In Memory Data Source

Grain supports in-memory data source for sequences that is sharable among mutiple processes.

It allows to avoid replicating the data in memory of each worker process.

Note: Currently this constrains storable values to only the int, float, bool, str (less than
10M bytes each), bytes (less than 10M bytes each), and None built-in data types.

```{code-cell}
---
executionInfo:
  elapsed: 22
  status: ok
  timestamp: 1734464322747
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: -0hJjr9UuNxR
outputId: bd2e4198-9902-4003-f639-58020fa953a1
---
in_memory_datasource = grain.InMemoryDataSource(range(5))

print(in_memory_datasource)
print(f"First Record Read: {in_memory_datasource[0]}")
```

```{code-cell}
---
executionInfo:
  elapsed: 21210
  status: ok
  timestamp: 1734464345506
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: 7aVz3km3_qAw
outputId: 349df7eb-e464-4164-ed0d-1dd683daf148
---
data_loader = grain.DataLoader(
    data_source=in_memory_datasource,
    sampler=grain.IndexSampler(
      num_records=len(in_memory_datasource),
      num_epochs=2,
      shard_options=grain.NoSharding(),
      shuffle=True,
      seed=0),
    worker_count=5)

data_iter = iter(data_loader)

for i in range(len(in_memory_datasource) * 2):
  x = next(data_iter)
  print(x)
```

+++ {"id": "lmRP3I2PCHTn"}

### Per-worker `ReadOptions`

You can configure per-worker data source read options (for example, number of threads, prefetch buffer size) in `ReadOptions`.

```{code-cell}
:id: _W360gEYNLYW

# Following configuration makes 8*10=80 threads reading data.
data_loader = grain.DataLoader(
    data_source=source,
    sampler=grain.IndexSampler(
      num_records=len(source),
      num_epochs=2,
      shard_options=grain.NoSharding(),
      shuffle=True,
      seed=0),
    worker_count=10,
    read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=500))
```
