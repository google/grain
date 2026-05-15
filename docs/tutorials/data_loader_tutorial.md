---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

+++ {"id": "qGiXX-sg4l9o"}

# `DataLoader` guide

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/data_loader_tutorial.ipynb)

+++ {"id": "rKvQbK6LK7Yd"}

## Install Grain and tutorial dependencies

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: EZ9EXOZKehes
outputId: 2de6bc20-8edb-4376-e5d3-3ec7bff35462
---
!pip install grain tfds-nightly opencv-python matplotlib orbax-checkpoint
import cv2
import orbax.checkpoint as ocp
import grain.python as grain
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

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qk4MtXvXe1Bl
outputId: 0ecd27db-9080-4a78-d13a-36b1ca590b77
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

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: y6Yav1dCg8Ny
outputId: 084e6c0b-79c5-4e03-8731-a9966eccbd65
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

```{code-cell} ipython3
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

```{code-cell} ipython3
:id: RhcG6Ehs9uVy

data_loader = grain.DataLoader(
    data_source=source,
    operations=transformations,
    sampler=index_sampler,
    worker_count=0)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: aXN-iffE94Qc
outputId: cae6e071-0964-4454-ce4c-0e66a1d4bf32
---
for element in data_loader:
  display(element["image"])
```

+++ {"id": "eODdpQcZeOfo"}

## Checkpointing
We provide `PyGrainCheckpointHandler` to checkpoint the iterator returned by Grain. It is recommended to use it with [Orbax](https://orbax.readthedocs.io), which can checkpoint both input pipeline and model and handles the edge cases for distributed training.

+++ {"id": "0JI0hvnUetWB"}

### Integration with Orbax
Orbax contains a powerful library for checkpointing various objects - incl. Flax models and Grain `DatasetIterators`. This makes it easy to checkpoint the `DatasetIterator` together with the model in a multihost environment. Orbax will take care of synchronizing all JAX processes.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: irJix4sJkNcf
outputId: 4dd5c594-176f-4d6e-9ae1-5f5b2532c472
---
data_iter = iter(data_loader)

num_steps = 5

# Read some elements.
for i in range(num_steps):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: qRoTuxBNl8DB
outputId: 1fd4cc3a-4cb4-4edd-f230-55631494b105
---
mngr = ocp.CheckpointManager("/tmp/orbax")

!rm -rf /tmp/orbax

# Save the checkpoint
assert mngr.save(
    step=num_steps, args=grain.PyGrainCheckpointSave(data_iter), force=True)
# Checkpoint saving in Orbax is asynchronous by default, so we'll wait until
# finished before examining checkpoint.
mngr.wait_until_finished()

!ls -R /tmp/orbax
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JVt1F3JbkWrz
outputId: 763d9e1c-61ad-4490-a2b0-6a0ff633a06b
---
!cat /tmp/orbax/*/*/*.json
```

+++ {"id": "5TFV1ScvsuTw"}

Note: the checkpoint contains the string representation of the sampler and the data source. Checkpoints are only valid when loaded with the same sampler/data source and Grain uses the string representation for a basic check.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: wT04O9cdlpPJ
outputId: 67d7ba4f-1bb1-4295-e304-8b0d677be7aa
---
# Read more elements and advance the iterator
for i in range(5, 10):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: Js3hheiGnykN
outputId: a47a427e-5ba5-4249-92c9-7614d42f752d
---
# Restore iterator from previously saved checkpoint
mngr.restore(num_steps, args=grain.PyGrainCheckpointRestore(data_iter))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: X6NT-drgoaJk
outputId: 785def82-0776-49e2-a0d4-9130fb01241a
---
# Iterator should be set back to start from 5.
for i in range(5, 10):
  x = next(data_iter)
  print(i, x["file_name"], x["label"])
```

+++ {"id": "btSRh4EL_Zbo"}

## Extras

+++ {"id": "2UJ_sreV9qWs"}

### Torchvision Dataset Source
If we look at `torchvision.datasets` (which we will shorten to `tvds`) the abstract class `tvds.Dataset` very almost matches the `data_sources.RandomAccessDataSource` protocol in `grain`. The only feature that might be missing is a `__len__` method (that returns the number of total elements in the data source). However, many of the included `tvds.Dataset` subclasses however provide this method, e.g. `tvds.CIFAR10`, which is what we use below.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: nbw8Tr95BepR
outputId: 99ba9c90-4343-4c2f-ed24-b9caf319071b
---
!pip install torchvision
import torchvision.datasets as tvds
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 258
id: FdPyI-dF9qWs
outputId: e0a4c0f6-eb19-4051-be77-4c58edd0a1d2
---
data_source = tvds.CIFAR10(root='./data', train=True, download=True)

def pil_to_numpy(example):
  # convert PIL images to numpy arrays, but can be torch tensors or anything
  image, label = example
  image = np.array(image)
  return image, label

# as before, grain.MapDataset, is an ordinary python iterator
data_loader = grain.MapDataset.source(data_source).map(pil_to_numpy)
data_loader = iter(data_loader)

for _ in range(4):
  image, label = next(data_loader)
  display(image)
```

+++ {"id": "LINDtlJY9qWs"}

#### Supplying `__len__`

In the event that you have a `tvds.Dataclass` which does not suply `__len__` one can be provided by patching it as follows.

```python
class UnfriendlyDataset(tvds.Dataset):
    def __getitem__(self, index: int):
        # some implementation here
        pass

def unfriendly_dataset_len(_):
    # determine the number of examples in the datset, lets say it's 10_000
    return 10_000

UnfriendlyDataset.__len__ = unfriendly_dataset_len
```
Then you can follow the steps above as we demonstrated with CIFAR10 and `grain.MapDataset`, one of the other grain data loaders.

+++ {"id": "LyIXwNWct7mQ"}

### In Memory Data Source

Grain supports in-memory data source for sequences that is sharable among mutiple processes.

It allows to avoid replicating the data in memory of each worker process.

Note: Currently this constrains storable values to only the int, float, bool, str (less than
10M bytes each), bytes (less than 10M bytes each), and None built-in data types.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: -0hJjr9UuNxR
outputId: 0aae5e52-601c-408a-ab19-fb58ad174e5a
---
in_memory_datasource = grain.InMemoryDataSource(range(5))

print(in_memory_datasource)
print(f"First Record Read: {in_memory_datasource[0]}")
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 7aVz3km3_qAw
outputId: e4fcef1d-c2f7-474a-ba4c-591e37c55c08
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

```{code-cell} ipython3
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
