---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.2
kernelspec:
  display_name: grain-dev
  language: python
  name: python3
---

+++ {"id": "jx-intro-md"}

# Plugging Grain into JAX training: batching + accelerator transfer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/jax_training_tutorial.ipynb)

This guide covers the last mile between a Grain pipeline and a JAX training step: how to **batch** records into arrays of the right shape, and how to **move those batches onto your accelerators** efficiently: host-device prefetch, sharding across devices, and distributed-training shards.

```{code-cell} ipython3
:id: jx-install

# @test {"output": "ignore"}
!pip install grain
# @test {"output": "ignore"}
!pip install tensorflow_datasets
```

```{code-cell} ipython3
:id: jx-imports

import grain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
```

+++ {"id": "jx-minimal-md"}

## 1. Minimal end-to-end pipeline

The shortest pipeline you'd want for JAX training: source -> shuffle -> preprocess -> **batch** -> iterate -> **`device_put`** -> step.

```{code-cell} ipython3
:id: jx-minimal-code

source = tfds.data_source("mnist", split="train")

ds = (
    grain.MapDataset.source(source)
    .seed(42)
    .shuffle()
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(batch_size=128, drop_remainder=True)  # new leading dim
    .to_iter_dataset()
)

for batch in ds:
    batch = jax.device_put(batch)  # default device
    print(jax.tree.map(lambda x: (x.shape, x.dtype), batch))
    break
```

+++ {"id": "jx-minimal-notes"}

A few things to notice:

- `batch(...)` lives on `MapDataset`. It stacks PyTree leaves along a **new leading axis** (here `[128, 28, 28, 1]` for images, `[128]` for labels).
- `drop_remainder=True` guarantees a static batch shape, which lets `jax.jit` cache one compiled version of the step.
- `to_iter_dataset()` turns the random-access `MapDataset` into an `IterDataset`. Do this **after** any random-access transforms (shuffle, batch, repeat) and **before** any streaming transforms (prefetch, `device_put`).

+++ {"id": "jx-batching-md"}

## 2. Batching tips that matter for JAX

**Stable shapes.** JAX recompiles whenever input shapes change. Pair `batch(drop_remainder=True)` with `.repeat()` so the loop never produces a short final batch:

```{code-cell} ipython3
:id: jx-repeat-code

ds = (
    grain.MapDataset.source(source)
    .seed(42)
    .shuffle()
    .repeat()  # infinite stream
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(128, drop_remainder=True)
)
print("length:", len(ds))  # sys.maxsize
```

+++ {"id": "jx-collate-md"}

**Custom collation.** The default `batch_fn` stacks leaves with `np.stack`. Pass your own when you need padding, ragged handling, or anything non-uniform:

```{code-cell} ipython3
:id: jx-collate-code

def pad_collate(items):
    max_len = max(x["tokens"].shape[0] for x in items)
    tokens = np.stack([
        np.pad(x["tokens"], (0, max_len - x["tokens"].shape[0]))
        for x in items
    ])
    return {"tokens": tokens}

# Toy stream of variable-length token sequences.
ragged = grain.MapDataset.source(
    [{"tokens": np.arange(np.random.randint(2, 6))} for _ in range(16)]
)
ragged = ragged.batch(4, batch_fn=pad_collate, drop_remainder=True)
for i in range(3):
    print(ragged[i]["tokens"])
```

+++ {"id": "jx-pad-md"}

For variable-length token streams, also look at `grain.experimental.batch_and_pad` — it pads partial final batches to the requested batch size with a sentinel, so you keep one static shape without dropping data.

+++ {"id": "jx-transfer-md"}

## 3. Moving batches to the accelerator

There are three options. Pick the lowest tier that meets your needs.

+++ {"id": "jx-option-a-md"}

### Option A: plain `jax.device_put`

Fine for prototyping and small models:

```{code-cell} ipython3
:id: jx-option-a-code

ds = (
    grain.MapDataset.source(source)
    .seed(42).shuffle()
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(128, drop_remainder=True)
    .to_iter_dataset()
)

for step, batch in zip(range(2), ds):
    batch = jax.device_put(batch)
    print(step, batch["image"].shape, batch["image"].sharding)
```

+++ {"id": "jx-option-a-caveat"}

The transfer happens on the main thread between every `next(...)`, so the host blocks while the device receives data. On a real training loop this can leave the accelerator idle.

+++ {"id": "jx-option-b-md"}

### Option B: overlap host work with `ThreadPrefetchIterDataset`

Run the pipeline's CPU work on a background thread so the next batch is ready by the time the device is done with the previous step:

```{code-cell} ipython3
:id: jx-option-b-code

ds = (
    grain.MapDataset.source(source)
    .seed(42).shuffle()
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(128, drop_remainder=True)
    .to_iter_dataset()
)
ds = grain.experimental.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=4)
ds = ds.map(jax.device_put)  # transfer still on iter thread

for step, batch in zip(range(3), ds):
    print(step, batch["image"].shape, batch["image"].sharding)
```

+++ {"id": "jx-option-c-md"}

### Option C: two-stage prefetch with `grain.experimental.device_put`

The recommended pattern for real training. It runs a CPU buffer **and** a device-resident buffer, so a batch is already on the accelerator before the step asks for it:

```{code-cell} ipython3
:id: jx-option-c-code

ds = (
    grain.MapDataset.source(source)
    .seed(42).shuffle()
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(128, drop_remainder=True)
    .to_iter_dataset()
)

ds = grain.experimental.device_put(
    ds=ds,
    device=jax.devices()[0],     # or a Sharding (see below)
    cpu_buffer_size=4,           # batches buffered on host
    device_buffer_size=2,        # batches buffered on device
)

for step, batch in zip(range(2), ds):
    # `batch` is already a jax.Array on-device.
    print(step, batch["image"].sharding)
```

+++ {"id": "jx-option-c-impl"}

Under the hood this is just `ThreadPrefetch -> map(jax.device_put) -> ThreadPrefetch`.

+++ {"id": "jx-shard-arrays-md"}

## 4. Multi-device: sharding a batch across accelerators

For data-parallel training across all local devices, pass a `Sharding` to `device_put` instead of a single device. Each batch is split along its first axis:

```{code-cell} ipython3
:id: jx-shard-arrays-code

devices = jax.devices()
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("data",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

ds = (
    grain.MapDataset.source(source)
    .seed(42).shuffle().repeat()
    .map(lambda r: {"image": r["image"].astype(np.float32) / 255.0,
                    "label": r["label"]})
    .batch(128, drop_remainder=True)
    .to_iter_dataset()
)

ds = grain.experimental.device_put(
    ds=ds,
    device=sharding,
    cpu_buffer_size=4,
    device_buffer_size=2,
)

for step, batch in zip(range(3), ds):
    print(step, batch["image"].sharding)
```

+++ {"id": "jx-shard-arrays-notes"}

Make sure `batch_size` is divisible by `len(devices)` — otherwise the sharding split fails. Inside your train step, decorate with `jax.jit` and JAX will compile a single SPMD program that handles the per-device slices automatically.

+++ {"id": "jx-template-md"}

## 5. Putting it all together

A realistic single-host, multi-device template:

```{code-cell} ipython3
:id: jx-template-code

BATCH_SIZE = 256
devices = jax.devices()
mesh = jax.sharding.Mesh(np.array(devices), axis_names=("data",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

def preprocess(r):
    return {"image": r["image"].astype(np.float32) / 255.0,
            "label": r["label"]}

ds = (
    grain.MapDataset.source(source)
    .seed(42).shuffle().repeat()
    .map(preprocess)
    .batch(BATCH_SIZE, drop_remainder=True)
    .to_iter_dataset()
)

ds = grain.experimental.device_put(
    ds=ds, device=sharding,
    cpu_buffer_size=4, device_buffer_size=2,
)

@jax.jit
def train_step(params, batch):
    # Replace with your real loss/update.
    return params + batch["image"].mean()

params = jnp.zeros(())
for step, batch in zip(range(3), ds):
    params = train_step(params, batch)
print("final params:", params)
```
