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
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Data sources
data_sources/protocol
Bagz <tutorials/data_sources/bagz_data_source_tutorial>
ArrayRecord <tutorials/data_sources/arrayrecord_data_source_tutorial>
Parquet <tutorials/data_sources/parquet_dataset_tutorial>
TfRecord <https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.TFRecordIterDataset.html#grain.experimental.TFRecordIterDataset>
TFDS <tutorials/dataset_advanced_tutorial>
HuggingFace <tutorials/data_sources/huggingface_dataset_tutorial>
PyTorch <tutorials/data_sources/pytorch_dataset_tutorial>
GCS <tutorials/data_sources/load_from_gcs_tutorial>
Amazon S3 <tutorials/data_sources/load_from_s3_tutorial>
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: Dataset
Basics <tutorials/dataset_basic_tutorial>
Advanced usage <tutorials/dataset_advanced_tutorial>
Transformations <https://google-grain.readthedocs.io/en/latest/grain.dataset.html>
Performance debugging <tutorials/dataset_debugging_tutorial>
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: DataLoader
data_loader/samplers
data_loader/transformations
Tutorial <tutorials/data_loader_tutorial>
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: API reference
grain
changelog
```

``` {toctree}
:maxdepth: 1
:hidden:
:caption: For contributors
behind_the_scenes
CONTRIBUTING
```