---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Loading and transforming HuggingFace datasets

HuggingFace (HF) platform provides a wide variety of ML models, datasets, and transformers for the worldwide community.
An easy access to these assets is guaranteed thanks to Python packages such as [datasets](https://pypi.org/project/datasets/) or [transformers](https://pypi.org/project/transformers/), available on PyPI.

In this tutorial you will learn how to utilize HF datasets and tools with Grain: How to load HF datasets and how to use HF transformers in your Grain pipeline.

```{code-cell} ipython3
# @test {"output": "ignore"}
!pip install grain
# @test {"output": "ignore"}
!pip install numpy datasets transformers
```

```{code-cell} ipython3
# Python standard library imports
from pprint import pprint
from dateutil.parser import parse

import grain
import numpy as np

# HF imports
from datasets import load_dataset
from transformers import AutoTokenizer
```

Let's first import an HF dataset. For the sake of simplicity let's proceed with [lhoestq/demo1](https://huggingface.co/datasets/lhoestq/demo1) - a minimal dataset comprised of five rows and six columns.

```{code-cell} ipython3
hf_dataset = load_dataset("lhoestq/demo1")
hf_train, hf_test = hf_dataset["train"], hf_dataset["test"]
hf_dataset
```

Each sample is a Python dictionary with string or integer data.

```{code-cell} ipython3
hf_train[0]
```

Now we define our data source class. HF dataset provides a random access, so we inherit from `RandomAccessDataSource` and implement required functions.

```{code-cell} ipython3
class HFSource(grain.sources.RandomAccessDataSource):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return len(self._dataset)
```

```{code-cell} ipython3
hf_train_src = HFSource(hf_train)
hf_test_src = HFSource(hf_test)
```

Let's assume that for our preprocessing pipeline we want the string `date` field to become a timestamp and the whole sample - a NumPy array.

```{code-cell} ipython3
def process_date(sample: dict) -> dict:
    sample["date"] = parse(sample["date"]).timestamp()
    return sample

def process_sample_to_np(sample: dict) -> np.ndarray:
    return np.array([*sample.values()], dtype=object)
```

Building a pipeline is as simple as chaining `map` calls. The resulting object is of type `MapMapDataset` with random access support.

```{code-cell} ipython3
pipeline_1 = (
    grain.MapDataset.source(hf_train_src)
    .shuffle(seed=10)  # shuffles globally
    .map(process_date)  # maps each element
    .map(process_sample_to_np)  # maps each element
)
```

```{code-cell} ipython3
list(pipeline_1)
```

Next we would like to tokenize the `review` field. LLM models operate on integers (encoded words) rather than raw strings. `AutoTokenizer` generic class ships `from_pretrained` method - accessor to models and tokenizers hosted on HF services.

Let's use `bert-base-uncased`, a case-insensitive BERT-based transformers model.

```{code-cell} ipython3
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer
```

Transforming a single review string yields a dictionary with three keys. We're only interested in `input_ids` since that is the encoded review.

```{code-cell} ipython3
review = hf_train[0]["review"]
pprint(review)
print("\n", tokenizer(review).keys(), "\n")
pprint(tokenizer(review)["input_ids"])
```

Plugging the selected transformer is as easy as before. We implement the `process_transformer` function and pass it to the `map` method.

```{code-cell} ipython3
def process_transformer(sample: dict) -> dict:
    sample["review"] = np.array(tokenizer(sample["review"])["input_ids"])
    return sample

pipeline_2 = (
    grain.MapDataset.source(hf_train)
    .shuffle(seed=10)
    .map(process_date)
    .map(process_transformer)
)
```

Now samples are less human- but more machine-friendly.

```{code-cell} ipython3
pipeline_2[1]
```

Time to build our final pipeline! The pipeline doesn't need to be restricted to `shuffle` and `map`. Grain has a rich API and hands us multiple functionalities such as: `filter`, `random_map`, `repeat`.

On top of the transformer we want to discard reviews that are rated three stars or less. It's crucial to mention that filtering changes the number of samples in the following steps so random access is no longer available. To perform `batching` as the final step we plug `.to_iter_dataset()` converting `MapDataset` to `IterDataset` - a dataset that gives us an iterator-like interface.

```{code-cell} ipython3
pipeline_3 = (
    grain.MapDataset.source(hf_train)
    .shuffle(seed=10)
    .map(process_date)
    .map(process_transformer)
    .filter(lambda x: x["star"] > 3)  # filters samples
    .map(process_sample_to_np)
    .to_iter_dataset()
    .batch(batch_size=2)  # batches consecutive elements
)
```

With `IterDataset` we can use Python built-ins, `iter` and `next`, to interact with the dataset.

```{code-cell} ipython3
next(iter(pipeline_3))
```

And that's it! We ended up with a batch with processed date, tokenized review, and filtered rating.
