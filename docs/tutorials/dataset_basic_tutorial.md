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

+++ {"id": "BvnXLPI_2dNJ"}

# Installs PyGrain (OSS only)

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: sHOibn5Q2GRt
outputId: f4c3e5a6-56b8-47f1-c5a1-a25fd0c433b3
---
# @test {"output": "ignore"}
!pip install grain
```

+++ {"id": "8UuJxi2p3lPp"}

# Imports

```{code-cell}
---
executionInfo:
  elapsed: 3685
  status: ok
  timestamp: 1733956461547
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: ZgB5xOru2Zz8
---
import grain.python as grain
import pprint
```

+++ {"id": "gPv3wrQd3pZS"}

# `MapDataset`

`MapDataset` defines a dataset that supports efficient random access. Think of it as an (infinite) `Sequence` that computes values lazily. It will either be the starting point of the input pipeline or in the middle of the pipeline following another `MapDataset`. Grain provides many basic transformations for users to get started.

```{code-cell}
---
executionInfo:
  elapsed: 55
  status: ok
  timestamp: 1733956464592
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: 3z3Em5jC2iVz
outputId: a7f3963f-6493-4132-e3f6-17e0691562e8
---
dataset = (
    # You can also use a shortcut grain.MapDataset.range for
    # range-like input.
    grain.MapDataset.source([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    .shuffle(seed=10)  # Shuffles globally.
    .map(lambda x: x+1)  # Maps each element.
    .batch(batch_size=2)  # Batches consecutive elements.
)

pprint.pprint(dataset[0])
pprint.pprint(list(dataset))
```

+++ {"id": "Aii_JDBw5SEI"}

The requirement for `MapDataset`'s source is a `grain.RandomAccessDataSource` interface: i.e. `__getitem__` and `__len__`.

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1733956476197
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: kCbDSzlS4a-A
---
# Note: Inheriting `grain.RandomAccessDataSource` is optional but recommended.
class MySource(grain.RandomAccessDataSource):
  def __init__(self):
    self._data = [0, 1, 2, 3, 4, 5, 6, 7]

  def __getitem__(self, idx):
    return self._data[idx]

  def __len__(self):
    return len(self._data)
```

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1733956478654
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: m8Cyn7gt6FYy
outputId: 35a47622-ee59-44a1-cd81-3a1e41b8ca74
---
source = MySource()

dataset = (
    grain.MapDataset.source(source)
    .shuffle(seed=10)  # Shuffles globally.
    .map(lambda x: x+1)  # Maps each element.
    .batch(batch_size=2)  # Batches consecutive elements.
)

pprint.pprint(dataset[0])
pprint.pprint(list(dataset))
```

+++ {"id": "zKv2kWjB6XPd"}

Access by index will never raise an `IndexError` and can treat indices that are equal or larger than the length as a different epoch (e.g. shuffle differently, use different random numbers).

```{code-cell}
---
executionInfo:
  elapsed: 53
  status: ok
  timestamp: 1733956481133
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: GSW1cJe06NEO
outputId: 080bb98b-5184-469a-97be-f74416568449
---
# Prints the 3rd element of the second epoch.
pprint.pprint(dataset[len(dataset)+2])
```

+++ {"id": "azfAr8F37njE"}

Note that `dataset[idx] == dataset[len(dataset) + idx]` iff there's no random transfomations. Since `dataset` has global shuffle, different epochs are shuffled differently:

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1733956483414
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: _o3wxb8k7XDY
outputId: 17f2b8ab-128e-41dc-ace0-ca3d32162c3a
---
pprint.pprint(dataset[len(dataset)+2] == dataset[2])
```

+++ {"id": "B2kLX0fa8GfV"}

You can use `filter` to remove elements not needed but it will return `None` to indicate that there is no element at the given index.

Returning `None` for the majority of positions can negatively impact performance of the pipeline. For example, if your pipeline filters 90% of the data it might be better to store a filtered version of your dataset.

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1733956489906
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: ai4zcltV7sSN
outputId: f9c0bd67-12c0-498a-cf41-6b2aa98fff5b
---
filtered_dataset = dataset.filter(lambda e: (e[0] + e[1]) % 2 == 0)

pprint.pprint(f"Length of this dataset: {len(filtered_dataset)}")
pprint.pprint([filtered_dataset[i] for i in range(len(filtered_dataset))])
```

+++ {"id": "FJLK_BQj9GuG"}

`MapDataset` also supports slicing using the same syntax as Python lists. This returns a `MapDataset` representing the sliced section. Slicing is the easiest way to "shard" data during distributed training.

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1733956492078
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: -fuS_OGS8x5Z
outputId: 630a9277-d6b9-419c-dc9a-78a81bf63688
---
shard_index = 0
shard_count = 2

sharded_dataset = dataset[shard_index::shard_count]
print(f"Sharded dataset length = {len(sharded_dataset)}")
pprint.pprint(sharded_dataset[0])
pprint.pprint(sharded_dataset[1])
```

+++ {"id": "KvycxocM-Fpk"}

For the actual running training with the dataset, we should convert `MapDataset` into `IterDataset` to leverage parallel prefetching to hide the latency of each element's IO using Python threads.

This brings us to the next section of the tutorial: `IterDataset`.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1733956494196
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: FnWPIpce9aAJ
outputId: 495c2c20-7de8-4ee9-e1d8-a21b30e5aa77
---
iter_dataset = sharded_dataset.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=500))

for element in iter_dataset:
  pprint.pprint(element)
```

+++ {"id": "W-Brm4Mh_Bo1"}

# IterDataset

Most data pipelines will start with one or more `MapDataset` (often derived from a `RandomAccessDataSource`) and switch to `IterDataset` late or not at all. `IterDataset` does not support efficient random access and only supports iterating over it. It's an `Iterable`.

Any `MapDataset` can be turned into a `IterDataset` by calling `to_iter_dataset`. When possible this should happen late in the pipeline since it will restrict the transformations that can come after it (e.g. global shuffle must come before). This conversion by default skips `None` elements.

+++ {"id": "GDO1u2tQ_zPz"}

`DatasetIterator` is a stateful iterator of `IterDataset`. The state of the iterator can be cheaply saved and restored. This is intended for checkpointing the input pipeline together with the trained model. The returned state will not contain data that flows through the pipeline.

Essentially, `DatasetIterator` only checkpoints index information for it to recover (assuming the underlying content of files will not change).

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1733956497535
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: DRgatGFX_nxL
outputId: 89383832-1676-4f16-ebfa-bb74d913d415
---
dataset_iter = iter(dataset)
pprint.pprint(isinstance(dataset_iter, grain.DatasetIterator))
```

```{code-cell}
---
executionInfo:
  elapsed: 157
  status: ok
  timestamp: 1733956500621
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: dOCiJfSJ_vi4
outputId: 01d5a3e8-4d43-4818-fe5c-184d8afda412
---
pprint.pprint(next(dataset_iter))

checkpoint = dataset_iter.get_state()

pprint.pprint(next(dataset_iter))

# Recover the iterator to the state after the first produced element.
dataset_iter.set_state(checkpoint)

pprint.pprint(next(dataset_iter))  # This should generate the same element as above
```
