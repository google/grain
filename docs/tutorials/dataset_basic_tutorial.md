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

+++ {"id": "JBD-hzAdSv7F"}

# `Dataset` basics

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/dataset_basic_tutorial.ipynb)

`Dataset` is a low-level API that uses chaining syntax to define data
transformation steps. It allows more general types of processing (e.g. dataset
mixing) and more control over the execution (e.g. different order of data
sharding and shuffling). `Dataset` transformations are composed in a way that
allows to preserve random access property past the source and some of the
transformations. This, among other things, can be used for debugging by
evaluating dataset elements at specific positions without processing the entire
dataset.

There are 3 main classes comprising the `Dataset` API: `MapDataset`,
`IterDataset`, and `DatasetIterator`. Most data pipelines will start with one or
more `MapDataset` (often derived from a `RandomAccessDataSource`) and switch to
`IterDataset` late or not at all. The following sections will provide more
details about each class.

+++ {"id": "BvnXLPI_2dNJ"}

## Install and import Grain

```{code-cell}
:id: sHOibn5Q2GRt

# @test {"output": "ignore"}
!pip install grain
```

```{code-cell}
---
executionInfo:
  elapsed: 4007
  status: ok
  timestamp: 1734118718461
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: FCZXw2YhhPyu
---
import grain.python as grain
import pprint
```

+++ {"id": "gPv3wrQd3pZS"}

## `MapDataset`

`MapDataset` defines a dataset that supports efficient random access. Think of it as an (infinite) `Sequence` that computes values lazily. It will either be the starting point of the input pipeline or in the middle of the pipeline following another `MapDataset`. Grain provides many basic transformations for users to get started.

```{code-cell}
---
executionInfo:
  elapsed: 57
  status: ok
  timestamp: 1734118721729
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: 3z3Em5jC2iVz
outputId: c63e44a8-7a03-4d01-c210-b292ad6c5bdf
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
  elapsed: 55
  status: ok
  timestamp: 1734118753268
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: 592ut9AgiDCz
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
  elapsed: 5
  status: ok
  timestamp: 1734118755899
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: m8Cyn7gt6FYy
outputId: ff59f222-042c-48fb-d426-c2b3da9a2017
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
  elapsed: 3
  status: ok
  timestamp: 1734118760614
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: GSW1cJe06NEO
outputId: ffe2b9e8-069c-45f1-ac93-c391604b5b34
---
# Prints the 3rd element of the second epoch.
pprint.pprint(dataset[len(dataset)+2])
```

+++ {"id": "azfAr8F37njE"}

Note that `dataset[idx] == dataset[len(dataset) + idx]` iff there's no random transfomations. Since `dataset` has global shuffle, different epochs are shuffled differently:

```{code-cell}
---
executionInfo:
  elapsed: 3
  status: ok
  timestamp: 1734118766095
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: _o3wxb8k7XDY
outputId: f4c2a263-0084-45d3-dd0f-51f58c96bead
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
  timestamp: 1734118794030
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: ai4zcltV7sSN
outputId: c8818ad2-c7d7-414f-8359-0bd2e679b9ed
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
  elapsed: 57
  status: ok
  timestamp: 1734118798792
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: -fuS_OGS8x5Z
outputId: 7ce27bed-0adb-43a4-8abb-8d48937f1e22
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
  elapsed: 55
  status: ok
  timestamp: 1734118801247
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: FnWPIpce9aAJ
outputId: dba2951e-a965-4dd3-816c-dcbbea6352f7
---
iter_dataset = sharded_dataset.to_iter_dataset(grain.ReadOptions(num_threads=16, prefetch_buffer_size=500))

for element in iter_dataset:
  pprint.pprint(element)
```

+++ {"id": "W-Brm4Mh_Bo1"}

## IterDataset

Most data pipelines will start with one or more `MapDataset` (often derived from a `RandomAccessDataSource`) and switch to `IterDataset` late or not at all. `IterDataset` does not support efficient random access and only supports iterating over it. It's an `Iterable`.

Any `MapDataset` can be turned into a `IterDataset` by calling `to_iter_dataset`. When possible this should happen late in the pipeline since it will restrict the transformations that can come after it (e.g. global shuffle must come before). This conversion by default skips `None` elements.

+++ {"id": "GDO1u2tQ_zPz"}

`DatasetIterator` is a stateful iterator of `IterDataset`. The state of the iterator can be cheaply saved and restored. This is intended for checkpointing the input pipeline together with the trained model. The returned state will not contain data that flows through the pipeline.

Essentially, `DatasetIterator` only checkpoints index information for it to recover (assuming the underlying content of files will not change).

```{code-cell}
---
executionInfo:
  elapsed: 57
  status: ok
  timestamp: 1734118805719
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: DRgatGFX_nxL
outputId: 5ec46759-41a9-4211-c856-3f46c2ee2a9c
---
dataset_iter = iter(dataset)
pprint.pprint(isinstance(dataset_iter, grain.DatasetIterator))
```

```{code-cell}
---
executionInfo:
  elapsed: 184
  status: ok
  timestamp: 1734118814192
  user:
    displayName: ''
    userId: ''
  user_tz: 480
id: dOCiJfSJ_vi4
outputId: 6e010a76-11e3-4aad-ef16-93c00aa6ae27
---
pprint.pprint(next(dataset_iter))

checkpoint = dataset_iter.get_state()

pprint.pprint(next(dataset_iter))

# Recover the iterator to the state after the first produced element.
dataset_iter.set_state(checkpoint)

pprint.pprint(next(dataset_iter))  # This should generate the same element as above
```
