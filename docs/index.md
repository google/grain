# Grain - Feeding JAX Models

Grain is a library for reading data for training and evaluating JAX models. It's
open source, fast and deterministic.

PyGrain is the pure Python backend for Grain, primarily targeted at JAX users.

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} Powerful
Users can bring arbitrary python transformations.
:::

:::{grid-item-card} Flexible
PyGrain is designed in a modular fashion. Users can easily
override PyGrain components if need be with their own implementation.
:::

:::{grid-item-card} Deterministic
 Multiple runs of the same pipeline should produce the same
output.
:::

:::{grid-item-card} Resilient to preemptions
PyGrain is designed such that
checkpoints have minimal size. After pre-emption, PyGrain can resume from where
it left off and produce the same output as if it was never preempted.
:::

:::{grid-item-card} Performant
We took care while designing PyGrain to ensure that it's
performant (refer to the[Behind the Scenes](behind_the_scenes.md)
section of the documentation.)
We also tested it against multiple data modalities (e.g.Text/Audio/Images/Videos).
:::

:::{grid-item-card} With minimal dependencies
PyGrain should minimize its set
of dependencies when possible. For example, it should not depend on TensorFlow.
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
