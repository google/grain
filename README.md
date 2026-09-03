# Grain - Feeding JAX Models

[![Continuous integration](https://github.com/google/grain/actions/workflows/tests.yml/badge.svg)](https://github.com/google/grain/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/grain)](https://pypi.org/project/grain/)

[**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Reference docs**](https://google-grain.readthedocs.io/en/latest/)
| [**Change logs**](https://google-grain.readthedocs.io/en/latest/changelog.html)

Grain is a Python library for reading and processing data for training and
evaluating JAX models. It is flexible, fast and deterministic.

Grain allows to define data processing steps in a simple declarative way:

```python
import grain

dataset = (
    grain.MapDataset.source([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    .shuffle(seed=42)  # Shuffles elements globally.
    .map(lambda x: x+1)  # Maps each element.
    .batch(batch_size=2)  # Batches consecutive elements.
)

for batch in dataset:
  # Training step.
```

Grain is designed to work with JAX models but it does not require JAX to run
and can be used with other frameworks as well.

## Installation

Grain is available on [PyPI](https://pypi.org/project/grain/) and can be
installed with `pip install grain`.

### Supported platforms

Grain does not directly use GPU or TPU in its transformations, the processing
within Grain will be done on the CPU by default.

|         |  Linux  |   Mac   | Windows |
|---------|---------|---------|---------|
| x86_64  | yes     | no      | yes     |
| aarch64 | yes     | yes     | n/a     |

## Quickstart

- [Basic `Dataset` tutorial](https://google-grain.readthedocs.io/en/latest/tutorials/dataset_basic_tutorial.html)

### Dynamic Performance Autotuning

Grain supports automatic runtime tuning of worker threads and prefetch buffer
sizes to maximize pipeline throughput within specified memory and CPU budgets:

```python
import grain.python as grain

# 1. Define dataset transformation pipeline
dataset = (
    grain.MapDataset.range(10_000)
    .shuffle(seed=42)
    .map(lambda x: x + 1)
)

# 2. Configure prefetch with autotuned parameters
read_options = grain.ReadOptions(
    prefetch_buffer_size=grain.experimental.AutotuneParameter(
        name="buffer_size", initial_value=2, min_value=1, max_value=64
    ),
    num_threads=grain.experimental.AutotuneParameter(
        name="concurrency", initial_value=1, min_value=1, max_value=16
    ),
)
dataset = dataset.to_iter_dataset(read_options)

# 3. Enable online autotuning (Universal Scalability Law model)
model_config = grain.experimental.AutotuneModelConfig(
    ram_budget_gb=8.0,
    warmup_steps=50,
)
dataset = grain.experimental.autotune(dataset, model_config=model_config)

# 4. Iterate normally -- Grain dynamically optimizes parameters
for batch in dataset:
  # Training step.
  pass
```

## Citing Grain

To cite this repository:

```
@software{grain2023github,
  author = {Marvin Ritter and Ihor Indyk and Aayush Singh and Andrew Audibert and Anoosha Seelam and Camelia Hanes and Eric Lau and Jacek Olesiak and Jiyang Kang and Xihui Wu},
  title = {{Grain} - Feeding JAX Models},
  url = {http://github.com/google/grain},
  version = {0.2.12},
  year = {2023},
}
```

The version number is intended to be that from [pyproject.toml](https://github.com/google/grain/blob/main/pyproject.toml), and the year corresponds to the project's open-source release.

## Existing users

Grain is used by [MaxText](https://github.com/google/maxtext/tree/main),
[Gemma](https://github.com/google-deepmind/gemma),
[kauldron](https://github.com/google-research/kauldron),
[maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) and multiple
internal Google projects.
