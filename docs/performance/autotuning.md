# Grain - Autotuning

.md





## Overview

Grain supports performance autotuning for your input pipelines. Instead of
manually tuning prefetching buffer sizes and thread counts, `autotune`
dynamically adjusts these parameters during training to optimize throughput
while respecting memory/CPU budgets.

## Setup and Usage

To use autotuning, wrap your existing `IterDataset` with
`experimental.autotune()`. Set `allow_unknown_nodes=True` to gracefully handle
dataset transformations not explicitly profiled by autotune.

```python
import grain

# Create a normal dataset mapping and setup a thread-based prefetch with
# Autotune Parameters.
ds = grain.MapDataset.range(1000).map(lambda x: x + 1)
read_options = grain.ReadOptions(
    prefetch_buffer_size=grain.experimental.AutotuneParameter(
        name="buffer_size", initial_value=5, min_value=1, max_value=20
    ),
    num_threads=grain.experimental.AutotuneParameter(
        name="concurrency", initial_value=4, min_value=1, max_value=16
    ),
)
ds = ds.to_iter_dataset(read_options)

# Wrap with autotune.
model_config = grain.experimental.AutotuneModelConfig(
    ram_budget_gb=8.0,
    optimization_frequency=100,
    warmup_steps=50,
)
ds = grain.experimental.autotune(
    ds, model_config=model_config, allow_unknown_nodes=True
)

# Iterate and fetch elements normally.
for element in ds:
  # Pipeline automatically adjusts buffer_size and thread counts
  # behind the scenes.
  pass
```

## Monitoring Pipeline Performance

To monitor the parameters autotune
    selects during training in open-source environments, query the exported
    Prometheus metrics:

-   `grain_autotune_parameters`: Exposes the currently chosen `buffer_size` and
    `concurrency` values.

-   `grain_autotune_node_throughput`: Details the throughput of individual
    pipeline stages.
