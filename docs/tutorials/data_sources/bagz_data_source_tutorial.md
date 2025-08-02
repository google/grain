---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: grain-dev
  language: python
  name: python3
---

# Using Bagz Files

This tutorial gives an overview of integrating [Bagz](https://github.com/google-deepmind/bagz/) file format into Grain pipeline. Bagz, an alternative to ArrayRecord, is a novel file format which supports per-record compression and fast index-based lookup. It can also integrate with Apache Beam, a feature that we're also going to present in this tutorial.

## Setup

First we need to make sure we have all required packages. We pin JAX's version as the latest Apache Beam doesn't support NumPy 2.0 yet.

```{code-cell} ipython3
%pip install grain bagz apache-beam jax==0.4.38
```

```{code-cell} ipython3
import grain
import bagz
from bagz.beam import bagzio
import numpy as np
import pathlib
import random
import apache_beam as beam
```

```{code-cell} ipython3
assert np.__version__[0] == "1", "Apache Beam requires NumPy<2"
```

## Creating and reading Bagz files

As Bagz format is record-based we can use a simple loop and `bagz.Writer` context manager to write our contents to the output file.

```{code-cell} ipython3
random.seed(42)

records = list(f"Record: {random.randint(100, 1000)}" for _ in range(40))

file = pathlib.Path("data.bagz")

with bagz.Writer(file) as writer:
    for rec in records:
        writer.write(rec)
```

Bagz supports random access, therefore we can lookup items by index, check length of the file, and slice it arbitrarily.

```{code-cell} ipython3
reader = bagz.Reader(file)

print(len(reader))

print(reader[10])

print(list(reader[5:15]))
```

## Grain pipeline with Bagz files

With random access in mind, we can now consume Bagz files in a Grain pipeline with `grain.MapDataset` class. Then applying any transformation is the same as with other sources, such as ArrayRecord files.

```{code-cell} ipython3
dataset = (
    grain.MapDataset.source(reader)
    .shuffle(seed=42)
    .map(lambda x: x.decode())  # move from bytes to strings
    .filter(lambda x: x[-1] != "6")  # let's filter out some files
    .map(lambda x: x.upper())  # and capitalize them
    .to_iter_dataset()
)
```

```{code-cell} ipython3
print(f"Filtered out: {len(reader) - len(list(dataset))} records.")

list(dataset)
```

## Apache Beam

Likewise ArrayRecord, Bagz package can also integrate with the Apache Beam library to build ETL pipelines. In the example below we construct a pipeline which consumes some in-memory list, performs simple transformations, and loads outputs to a Bagz file with a `bagzio` module. `@0` in the filename indicates that we don't want sharding for this pipeline. To learn more about sharding in Bagz, please see [Bagz docs](https://github.com/google-deepmind/bagz/tree/main?tab=readme-ov-file#sharding).

```{code-cell} ipython3
with beam.Pipeline() as pipeline:
  data = ["record1", "record2", "record3"]
  _ = (
      pipeline
      | 'CreateData' >> beam.Create(data)
      | 'Capitalize' >> beam.Map(lambda x: x.upper())
      | 'Encode' >> beam.Map(lambda x: x.encode())
      | 'WriteData' >> bagzio.WriteToBagz('beam_data@0.bagz')
  )
```

```{code-cell} ipython3
file = pathlib.Path("beam_data.bagz")
reader = bagz.Reader(file)
print(list(reader))
```

In this tutorial we've learned about Bagz format.
