# Grain - Feeding JAX Models



Grain is a library for reading data for training and evaluating JAX models. It's
open source, fast and deterministic.

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Powerful
Users can bring arbitrary Python transformations.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Flexible
Grain is designed to
be modular. Users can readily override Grain components if need be with their
own implementation.
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Deterministic
Multiple runs of the same pipeline will produce the same output.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Resilient to preemptions
Grain is designed such that checkpoints have minimal size. After
pre-emption, Grain can resume from where it left off and produce the same output
as if it was never preempted.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Performant
We took care while designing Grain to ensure that it's performant (refer to the
[Behind the Scenes](behind_the_scenes.md) section of the documentation.) We also
tested it against multiple data modalities (e.g.Text/Audio/Images/Videos).
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` With minimal dependencies
Grain minimizes its set of dependencies when possible. For example, it should
not depend on TensorFlow.
:::
::::

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Get started
installation
api_choice
data_sources
behind_the_scenes
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Data Loader
data_loader/samplers
data_loader/transformations
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Tutorials
tutorials/data_loader_tutorial
tutorials/dataset_basic_tutorial
tutorials/dataset_advanced_tutorial
tutorials/dataset_debugging_tutorial
```

<!-- Automatically generated documentation from docstrings -->
``` {toctree}
:maxdepth: 1
:hidden:
:caption: References
autoapi/index
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Contributor guides
CONTRIBUTING
```