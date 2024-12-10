# Grain - Feeding JAX Models

Grain is a library for reading data for training and evaluating JAX models. It's
open source, fast and deterministic.

PyGrain is the pure Python backend for Grain, primarily targeted at JAX users.

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Powerful
Users can bring arbitrary Python transformations.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Flexible
PyGrain is designed to be modular. Users can readily
override PyGrain components if need be with their own implementation.
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Deterministic
 Multiple runs of the same pipeline should produce the same
output.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Resilient to pre-emptions
PyGrain is designed such that checkpoints have minimal size.
After pre-emption, PyGrain can resume from where it left off and produce the same
output as if it was never pre-empted.
:::

:::{grid-item-card} {octicon}`sparkles-fill;1.5em;sd-mr-1` Performant
We took care while designing PyGrain to ensure that it's
performant (refer to the [Behind the Scenes](behind_the_scenes.md)
section of the documentation.)
We also tested it against multiple data modalities (e.g.Text/Audio/Images/Videos).
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` With minimal dependencies
PyGrain minimizes its set of dependencies when possible.
For example, it should not depend on TensorFlow.
:::

::::

```{toctree}
:maxdepth: 1
:caption: Getting started
installation
behind_the_scenes
data_sources
```

```{toctree}
:maxdepth: 1
:caption: Data Loader
data_loader/samplers
data_loader/transformations
```

```{toctree}
:maxdepth: 1
:caption: Tutorials
tutorials/dataset_basic_tutorial
```

```{toctree}
:maxdepth: 1
:caption: Contributor guides
CONTRIBUTING
```

<!-- Automatically generated documentation from docstrings -->
```{toctree}
:maxdepth: 1
:caption: References
autoapi/index
```
