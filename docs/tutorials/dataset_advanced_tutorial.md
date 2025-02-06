---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "9ufbgPooUPJr"}

# Advanced `Dataset` usage

If you decided to use `Dataset` APIs, there's a good chance you want to do one or more processing steps described in this section, especially if working on data ingestion for generative model training.

```{code-cell}
:id: OFw1tjvkP3wb

# @test {"output": "ignore"}
!pip install grain
# @test {"output": "ignore"}
!pip install tensorflow_datasets
```

```{code-cell}
:id: fwvOt8-cqcQn

import pprint
import grain.python as grain
import numpy as np
import tensorflow_datasets as tfds
```

+++ {"id": "0ur4szH9l5_H"}

## Checkpointing

We provide `GrainCheckpoint{Save|Restore}` to checkpoint the
`DatasetIterator`. It is recommended to use it with
[Orbax](https://orbax.readthedocs.io/en/latest/index.html), which can checkpoint
both, input pipeline and model, and handles the edge cases for distributed
training.

```{code-cell}
:id: Tf-4Ljd2l5_H

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
:id: PGn-eSYil5_H

import orbax.checkpoint as ocp

mngr = ocp.CheckpointManager("/tmp/orbax")

!rm -rf /tmp/orbax

# Save the checkpoint.
assert mngr.save(
    step=num_steps, args=grain.GrainCheckpointSave(ds_iter), force=True
)
# Checkpoint saving in Orbax is asynchronous by default, so we'll wait until
# finished before examining checkpoint.
mngr.wait_until_finished()

# @test {"output": "ignore"}
!ls -R /tmp/orbax
```

```{code-cell}
:id: F012QoCJl5_H

!cat /tmp/orbax/*/*/*.json
```

```{code-cell}
:id: HURK2viXl5_H

# Read more elements and advance the iterator.
for i in range(4, 8):
  x = next(ds_iter)
  print(i, x["label"])
```

```{code-cell}
:id: u92Vkn1Hl5_H

# Restore iterator from the previously saved checkpoint.
mngr.restore(num_steps, args=grain.GrainCheckpointRestore(ds_iter))
# Iterator should be set back to start from 4.
for i in range(4, 8):
  x = next(ds_iter)
  print(i, x["label"])
```

+++ {"id": "GfA_bctscNyV"}

## Mixing datasets

`Dataset` allows mixing multiple data sources with potentially different transformations. There's two different ways of mixing `Dataset`s: `MapDataset.mix` and `IterDataset.mix`. If the mixed `Datasets` are sparse (e.g. one of the mixture components needs to be filtered) use `IterDataset.mix`, otherwise use `MapDataset.mix`.

```{code-cell}
:id: e8ROZXhtwOx3

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
pprint.pprint(np.shape(ds[0]))
```

+++ {"id": "crR2FZ1Gf6-O"}

If filtering inputs to the mixture, use `IterDataset.mix`.

```{code-cell}
:id: DTmUbvK4r8T8

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
pprint.pprint(np.shape(next(iter(ds))))
```

+++ {"id": "8TKInCDc6GUH"}

### Multi-epoch training

Mixed dataset length is determined by a combination of the length of the shortest input dataset and mixing weights. This means that once the shortest component is exhausted the new epoch will begin and the remainder of other datasets is going to be discarded. This can be avoided by repeating inputs to the mixture.

```{code-cell}
:id: JqetaYR36GUH

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

``` {code-cell}
:id: M0kB6nUQlneY

source1 = tfds.data_source(name="pneumonia_mnist", split="train")
source2 = tfds.data_source(name="mnist", split="train")
ds1 = grain.MapDataset.source(source1).seed(42).shuffle().repeat()
ds2 = grain.MapDataset.source(source2).seed(43).shuffle().repeat()

ds = grain.MapDataset.mix([ds1, ds2], weights=[1, 2])
print(f"Mixed dataset length = {len(ds1)}")  # sys.maxsize
```
