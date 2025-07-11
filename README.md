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
| x86_64  | yes     | no      | no      |
| aarch64 | yes     | yes     | n/a     |

## Quickstart

- [Basic `Dataset` tutorial](https://google-grain.readthedocs.io/en/latest/tutorials/dataset_basic_tutorial.html)

## Existing users

Grain is used by [MaxText](https://github.com/google/maxtext/tree/main),
[Gemma](https://github.com/google-deepmind/gemma),
[kauldron](https://github.com/google-research/kauldron),
[maxdiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) and multiple internal
Google projects.
