---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "9ufbgPooUPJr"}

# Advanced `Dataset` usage

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/dataset_advanced_tutorial.ipynb)

If you decided to use `Dataset` APIs, there's a good chance you want to do one or more processing steps described in this section, especially if working on data ingestion for generative model training.

```{code-cell}
:id: OFw1tjvkP3wb

# @test {"output": "ignore"}
!pip install grain
# @test {"output": "ignore"}
!pip install tensorflow_datasets
```

```{code-cell}
---
executionInfo:
  elapsed: 4380
  status: ok
  timestamp: 1744147018391
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: fwvOt8-cqcQn
---
import grain
import numpy as np
import tensorflow_datasets as tfds
from pprint import pprint
```

+++ {"id": "0ur4szH9l5_H"}

## Checkpointing

We provide `Checkpoint{Save|Restore}` to checkpoint the
`DatasetIterator`. It is recommended to use it with
[Orbax](https://orbax.readthedocs.io/en/latest/index.html), which can checkpoint
both, input pipeline and model, and handles the edge cases for distributed
training.

```{code-cell}
---
executionInfo:
  elapsed: 3250
  status: ok
  timestamp: 1744147032891
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: Tf-4Ljd2l5_H
outputId: 367f81d5-4437-4d74-b22a-537039393921
---
ds = (
    grain.MapDataset.source(tfds.data_source("mnist", split="train"))
    .seed(seed=45)
    .shuffle()
    .to_iter_dataset()
)

num_steps = 4
ds_iter = iter(ds)

# Read some elements.
for i in range(num_steps):
  x = next(ds_iter)
  print(i, x["label"])
```

```{code-cell}
:id: Wb2y5VoTl5_H

# @test {"output": "ignore"}
!pip install orbax
```

```{code-cell}
---
executionInfo:
  elapsed: 106
  status: ok
  timestamp: 1744147062733
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: PGn-eSYil5_H
outputId: 7835b15f-5607-4f4c-b16a-22164f51365e
---
import orbax.checkpoint as ocp

mngr = ocp.CheckpointManager("/tmp/orbax")

!rm -rf /tmp/orbax

# Save the checkpoint.
assert mngr.save(
    step=num_steps, args=grain.checkpoint.CheckpointSave(ds_iter), force=True
)
# Checkpoint saving in Orbax is asynchronous by default, so we'll wait until
# finished before examining checkpoint.
mngr.wait_until_finished()

# @test {"output": "ignore"}
!ls -R /tmp/orbax
```

```{code-cell}
---
executionInfo:
  elapsed: 54
  status: ok
  timestamp: 1744147066136
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: F012QoCJl5_H
outputId: 82d01250-df6d-4398-deb6-d2c7cb9d301c
---
!cat /tmp/orbax/*/*/*.json
```

```{code-cell}
---
executionInfo:
  elapsed: 53
  status: ok
  timestamp: 1744147068255
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: HURK2viXl5_H
outputId: 1fc99ff2-968d-47f0-f863-78725481f8ae
---
# Read more elements and advance the iterator.
for i in range(4, 8):
  x = next(ds_iter)
  print(i, x["label"])
```

```{code-cell}
---
executionInfo:
  elapsed: 113
  status: ok
  timestamp: 1744147072103
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: u92Vkn1Hl5_H
outputId: 1a294554-acf1-4398-fa61-d7bb1240ae75
---
# Restore iterator from the previously saved checkpoint.
mngr.restore(num_steps, args=grain.checkpoint.CheckpointRestore(ds_iter))
# Iterator should be set back to start from 4.
for i in range(4, 8):
  x = next(ds_iter)
  print(i, x["label"])
```

+++ {"id": "GfA_bctscNyV"}

## Mixing datasets

`Dataset` allows mixing multiple data sources with potentially different transformations. There's two different ways of mixing `Dataset`s: `MapDataset.mix` and `IterDataset.mix`. If the mixed `Datasets` are sparse (e.g. one of the mixture components needs to be filtered) use `IterDataset.mix`, otherwise use `MapDataset.mix`.

```{code-cell}
---
executionInfo:
  elapsed: 1294
  status: ok
  timestamp: 1744147084144
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: e8ROZXhtwOx3
outputId: 8d297df8-137d-4d7a-f7fe-50e2f27774fc
---
tfds.core.DatasetInfo.file_format = (
    tfds.core.file_adapters.FileFormat.ARRAY_RECORD
)
# This particular dataset mixes medical images with hand written numbers,
# probably not useful but allows to illustrate the API on small datasets.
source1 = tfds.data_source(name="pneumonia_mnist", split="train")
source2 = tfds.data_source(name="mnist", split="train")
ds1 = grain.MapDataset.source(source1).map(lambda features: features["image"])
ds2 = grain.MapDataset.source(source2).map(lambda features: features["image"])
ds = grain.MapDataset.mix([ds1, ds2], weights=[0.7, 0.3])
print(f"Mixed dataset length = {len(ds)}")
pprint(np.shape(ds[0]))
```

+++ {"id": "crR2FZ1Gf6-O"}

If filtering inputs to the mixture, use `IterDataset.mix`.

```{code-cell}
---
executionInfo:
  elapsed: 1594
  status: ok
  timestamp: 1744147093681
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: DTmUbvK4r8T8
outputId: b1bbf184-5edb-49ce-bcae-01face49d199
---
source1 = tfds.data_source(name="pneumonia_mnist", split="train")
source2 = tfds.data_source(name="mnist", split="train")
ds1 = (
    grain.MapDataset.source(source1)
    .filter(lambda features: int(features["label"]) == 1)
    .to_iter_dataset()
)
ds2 = (
    grain.MapDataset.source(source2)
    .filter(lambda features: int(features["label"]) > 4)
    .to_iter_dataset()
)

ds = grain.IterDataset.mix([ds1, ds2], weights=[0.7, 0.3]).map(
    lambda features: features["image"]
)
pprint(np.shape(next(iter(ds))))
```

+++ {"id": "8TKInCDc6GUH"}

### Multi-epoch training

Mixed dataset length is determined by a combination of the length of the shortest input dataset and mixing weights. This means that once the shortest component is exhausted the new epoch will begin and the remainder of other datasets is going to be discarded. This can be avoided by repeating inputs to the mixture.

```{code-cell}
---
executionInfo:
  elapsed: 1154
  status: ok
  timestamp: 1744147102879
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: JqetaYR36GUH
outputId: 525d506c-b8c8-42bb-dc62-2f2aa0fea661
---
source1 = tfds.data_source(name="pneumonia_mnist", split="train")
source2 = tfds.data_source(name="mnist", split="train")
ds1 = grain.MapDataset.source(source1).repeat()
ds2 = grain.MapDataset.source(source2).repeat()

ds = grain.MapDataset.mix([ds1, ds2], weights=[1, 2])
print(f"Mixed dataset length = {len(ds1)}")  # sys.maxsize
print(f"Mixed dataset length = {len(ds2)}")  # sys.maxsize
# Ds1 and ds2 are repeated to fill out the sys.maxsize with respect to weights.
print(f"Mixed dataset length = {len(ds)}")  # sys.maxsize
```

+++ {"id": "aulM2cVQlneY"}

### Shuffling

If you need to globally shuffle the mixed data prefer shuffling individual
`Dataset`s before mixing. This will ensure that the actual weights of the mixed
`Dataset`s are stable and as close as possible to the provided weights.

Additionally, make sure to provide different seeds to different mixture
components. This way there's no chance of introducing a seed dependency between
the components if the random transformations overlap.

```{code-cell}
---
executionInfo:
  elapsed: 1096
  status: ok
  timestamp: 1744147114851
  user:
    displayName: ''
    userId: ''
  user_tz: 420
id: OTveP3UQE7xv
outputId: fe861177-11af-4a8c-aa22-b19bbb18e8b6
---
source1 = tfds.data_source(name="pneumonia_mnist", split="train")
source2 = tfds.data_source(name="mnist", split="train")
ds1 = grain.MapDataset.source(source1).seed(42).shuffle().repeat()
ds2 = grain.MapDataset.source(source2).seed(43).shuffle().repeat()

ds = grain.MapDataset.mix([ds1, ds2], weights=[1, 2])
print(f"Mixed dataset length = {len(ds)}")  # sys.maxsize
```
