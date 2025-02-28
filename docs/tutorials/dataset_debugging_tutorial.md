---
jupytext:
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "OHoxgqr6sRKE"}

# Performance & Debugging tool
Grain offers two configurable modes that can be set to gain deeper insights into
pipeline execution and identify potential issues.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/dataset_debugging_tutorial.ipynb)

```{code-cell}
:id: xw_-jT1r6zNM

# @test {"output": "ignore"}
!pip install grain
```

+++ {"id": "YLaRRlCPsRKE"}

## Visualization mode
To get an overview of your dataset pipeline structure and clear understanding of
how the data flows, enable visualization mode. This will log a visual
representation of your pipeline, allowing you to easily identify different
transformation stages and their relationships. To enable visualization mode, set
the flag `--grain_py_dataset_visualization_output_dir=""` or call
`grain.config.update("py_dataset_visualization_output_dir", "")`

```{code-cell}
:id: 4y89Wx7PsRKE

# @test {"output": "ignore"}
import grain.python as grain

grain.config.update("py_dataset_visualization_output_dir", "")
ds = (
    grain.MapDataset.range(20)
    .seed(seed=42)
    .shuffle()
    .batch(batch_size=2)
    .map(lambda x: x)
    .to_iter_dataset()
)
it = iter(ds)

# Visualization graph is constructed once the dataset produces the first element
for _ in range(10):
  next(it)
```

+++ {"id": "_3h-u2I1i7wv"}

## Debug mode
To troubleshoot performance issues in your dataset pipeline, enable debug mode.
This will log a real-time execution summary of the pipeline at one-minute
intervals. This execution summary provides a detailed information on each
transformation stage such as processing time, number of elements processed and
other details that helps in identifying the slower stages in the pipeline.
To enable debug mode, set the flag `--grain_py_debug_mode=true` or call
`grain.config.update("py_debug_mode",True)`

```{code-cell}
:id: bN45Z58E3jGS

import time


# Define a dummy slow preprocessing function
def _dummy_slow_fn(x):
  time.sleep(10)
  return x
```

```{code-cell}
---
colab:
  height: 897
id: bN45Z58E3jGS
outputId: f3d640a8-1eae-414f-e6eb-e7c02c9a91df
---
# @test {"output": "ignore"}
import time

grain.config.update("py_debug_mode", True)

ds = (
    grain.MapDataset.range(20)
    .seed(seed=42)
    .shuffle()
    .batch(batch_size=2)
    .map(_dummy_slow_fn)
    .to_iter_dataset()
    .map(_dummy_slow_fn)
)
it = iter(ds)

for _ in range(10):
  next(it)
```

+++ {"id": "eSu9SOP8_x6A"}

In the above execution summary, 86% of the time is spent in the
`MapDatasetIterator` node and is the slowest stage of the pipeline.

Note that although from the `total_processing_time`, it might appear that
`MapMapDataset`(id:2) is the slowest stage, nodes from the id 2 to 6 are
executed in multiple threads and hence, the `total_processing_time` of these
nodes should be compared to the `total_processing_time` of iterator nodes(id:0)
