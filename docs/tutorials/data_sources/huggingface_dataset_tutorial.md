---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Loading and transforming HuggingFace datasets

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/data_sources/huggingface_dataset_tutorial.ipynb)

HuggingFace (HF) platform provides a wide variety of ML models, datasets, and
transformers for the worldwide community. An easy access to these assets is
guaranteed thanks to Python packages such as
[datasets](https://pypi.org/project/datasets/) or
[transformers](https://pypi.org/project/transformers/), available on PyPI.

In this tutorial you will learn how to utilize HF datasets and tools with Grain:
How to load HF datasets and how to use HF transformers in your Grain pipeline.

+++

## Setup

To run the notebook you need to have a few packages installed in your
environment: `grain`, `numpy`, and Two HF packages: `datasets` and
`transformers`.

```{code-cell} ipython3
!pip install grain
!pip install -U numpy datasets transformers huggingface_hub fsspec
```

```{code-cell} ipython3
# Python standard library
from pprint import pprint
# HF imports
from datasets import load_dataset
from dateutil.parser import parse
import grain
import numpy as np
from transformers import AutoTokenizer
```

## Loading dataset

Let's first import an HF dataset. For the sake of simplicity let's proceed with
[lhoestq/demo1](https://huggingface.co/datasets/lhoestq/demo1) - a minimal
dataset comprised of five rows and six columns.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 475
  referenced_widgets: [7fd8697dd8364fb9a590856497fad1a0, f3562633c4014e4ca0e05a4ed3a3aa91,
    209bcc180fe94a67820c0c0cc46e0e14, 2674af4d08154968bf3faf0f15539cb9, 92ae6b9682e9495a8dfe08b51d6219bb,
    3324af7a3b50466488f8bda2f50c84ab, 43f91de405a34b479bf3fbf1568cdeee, a9ccdcaf294d445cb4b6b4e4d3370d79,
    cfd1e2ced0574ec2a97e7086247a9bb7, 73eca8ecbdcc4ef4a80615c15e28af9c, c68d5decc51444fbb47c0bcf050e6771,
    7a323981c2e241019e57847ad660e1dd, 0c17f3f6f7cd400ba0952b84a838b71e, f7990fed2950436997256f75bbf67572,
    3513f20a11864e3d95c1c8870fbc8bd7, 87d4ac0b069741958df8ab709fe94915, 7f9fed2344c34e548663756d211f5713,
    86634506a20a4c03b2971f7133b1fe33, bf5872b7a74c48ff9372afb6ec568cbd, 1412ae74fdcc4415862e0bc5824c1fe7,
    8a64ef7906424c6aaac1537c74062454, e6e1c27ec4ab4580b9e56c74e8cc557c, 9a248d5318ff4c11b050a8e6458b0ef4,
    3a4e781117be427b9738468273852be1, cfa00003efe14900a56754dc83e6db26, cdb7a96abf1845659992a6e22e7eb9aa,
    62a2904358b244e6aa42197973fb7ada, 32f5cead9f8646e692fb0dd122e1cd06, bb4c77b8c1dc43969bb072d509c67bc4,
    9664fefa9e2c4e9eb784f1ca462d16de, 3c5f905f85ff4b3b9b550ec093fd702b, 8b849f7d439c46f6bff186c026741a08,
    030fe23cefab4e50b99e09dd0d6994b5, d7a1922d53c145b0a8d23f76e99c8fbc, 4dbca008cde34ceabf953f8e178c3110,
    0f1e184fd885453da0a589767c8c9675, 87d0147e77c148d399b253a5e7409b2f, ff71fdc0b3de4c74b672f1d27120bea3,
    c8a51db11b0042129afc6081ab30ecd6, 12b4d1cf11c64ee782b94192a37f9069, 074a0fcb5cf9446fa54a34908a77b1c0,
    692392e7f569413e90d886731b16ec06, b0cf3077a36443efbf97e8e5da0b97ff, 8057dc43aa3847cd845fc3611ca67fcd,
    aeba1484c3be4b79b8d6c023c0ff3986, 0bd5608656ee4251859041361a82459e, 157ffa519bf343b0994103851f3d7553,
    6d249c505a604d458763838fbcff14f1, 3f4906b0bab44dfd801b6a1498069ee9, 8e439cc40cf646e2a80225ca22aa46fa,
    36235b5734084ad9899852b9942544b5, 3ffa056231d142c1bdb879d7c3fc4793, c795836fe9e94e12825a35e0bfaf7fd3,
    64d28aa9d870477a9958eba2a11b8ee2, a0f8e2ca3b8e4e909c02c925b629f558]
executionInfo: {}
outputId: 759aaf9b-a8a0-4e8f-b8ba-24ed532f138a
---
hf_dataset = load_dataset("lhoestq/demo1")
hf_train, hf_test = hf_dataset["train"], hf_dataset["test"]
hf_dataset
```

Each sample is a Python dictionary with string or integer data.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo: {}
outputId: e1e93cfd-c6dc-4f97-a0d9-4f8c4075662f
---
hf_train[0]
```

## Preprocessing

Let's assume that for our preprocessing pipeline we want the string `date` field
to become a timestamp and the sample values - NumPy arrays.

```{code-cell} ipython3
def process_date(sample: dict) -> dict:
  sample["date"] = parse(sample["date"]).timestamp()
  return sample


def process_sample_to_np(sample: dict) -> np.ndarray:
  for name, value in sample.items():
    sample[name] = np.asarray(value)
  return sample
```

Building a pipeline is as simple as chaining `map` calls. HF dataset supports
random access so we can pass it directly to a `source` method. The resulting
object is of type `grain.MapDataset` with random access support.

```{code-cell} ipython3
dataset = (
    grain.MapDataset.source(hf_train)
    .shuffle(seed=42)  # shuffles globally
    .map(process_date)  # maps each element
    .map(process_sample_to_np)  # maps each element
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo: {}
outputId: efaf729c-574e-40ce-eb43-7aeef009cc5e
---
list(dataset)
```

## Tokenizer

Next we would like to tokenize the `review` field. LLM models operate on
integers (encoded words) rather than raw strings. `AutoTokenizer` generic class
ships `from_pretrained` method - accessor to models and tokenizers hosted on HF
services.

Let's use `bert-base-uncased`, a case-insensitive BERT-based transformers model.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 319
  referenced_widgets: [8cb7d79a8e224bb1bd8852fcb11ff570, 6b5c2d21540a49fb99738e8e35edb1d9,
    39eda72991f8496184f130dd088de9eb, 4c7d00ff4c1645a9a724c0359c28e4af, 85e62f49ad12471ab036b8773bc660b4,
    023c4f23e1cc48249180450620ae2eed, aebab11823b6441eb8be8a4f9a12b19d, a6295cdd89254bcbb1de4311b5b4a74c,
    6f666ebd19c148889af1d61455c3991d, 01ba3f08658444df8f04cca28867cfeb, 9454bb4f785f4dd4928e387a26686871,
    6d669d817b7741ddb8edd93afce701fa, 2c92e6569c1e4f5095a36fef225f1681, 2d13f1a4b5d24607ae0773cba8c615ea,
    c7fce359fc8a41f6a38e71a491e892e6, de09d773c94e40f2b97256adaa7eb6a8, 64ad881635554958bc9aadb2d5603061,
    9b4ec1d52738491abfdc0b22c8f3d2ac, 57f4f4576b7b429882643ff11614876a, 40d6df3f70fb44eb874abcc76cd9273e,
    3d43c76cbf1b4e9b80ae34d4513e3422, 45e1776917dc40138f351b37339d2a3d, 59005917d3ac4af68a1438b671995f5c,
    e6b559dbc58f4974a84a319e35de1fea, f17ea66fcf22456b8430366e0a78069c, d938ee2e921e490e876692d8dc3406f9,
    dc564098b05a4c06b86f6d5b58d8f757, 5e1ca4e242764c729ee0f1d199356676, c40406fc91064288aa5d2813386ded05,
    37eae6ba68b44ad2a2c386088e2fe819, 6a5cad9d94ca434fbd4cd90338f83a70, 68bc9626cbfa4a7cb700bc57c5008d33,
    908ced8ad02f4c4295556b2ea08a2737, 3ed57ed762a2436ab54787981a072211, 26d3776473fb4f28a6ea45e4dc49cf1e,
    e80c0af69d2b42dda3b57317d64cd9a9, 6f9656ca5eb04292aefd328fedde9c5c, 774c370156bf4e34bcaf90e40d5180d6,
    db52bd757e824153a1aba045f7d463df, 39c9cfcae881420cb719dd47e93692c5, 7bc1f8fd0477488ba4207c144bfd94c4,
    9fde98ad78674b77af4456ff73b12150, 5ee5ce2e08ad49b0ab1576fd5fbdf22b, d10dcf01ca534f7697f557509bc7f5ca]
executionInfo: {}
outputId: 419c9c4c-ee10-495b-b580-d59f89ebbee1
---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer
```

Transforming a single review string yields a dictionary with three keys. We're
only interested in `input_ids` since that is the encoded review.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo: {}
outputId: ae574d6f-240a-4f85-cc07-d4918152cc27
---
review = hf_train[0]["review"]
pprint(review)
print("\n", tokenizer(review).keys(), "\n")
pprint(np.asarray(tokenizer(review)["input_ids"]))
```

Plugging the selected transformer is as easy as before. We implement the
`process_transformer` function and pass it to the `map` method.

Note that the tokenized reviews have different lengths, and accelerators such as GPUs and TPUs typically require static rectangular batch shapes. For simplicity in this tutorial we will pad them to the same length before batching. For advanced use cases please take a look at our example packing imlementations: [first-fit](https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.FirstFitPackIterDataset.html#grain.experimental.FirstFitPackIterDataset) and [concat-and-split](https://google-grain.readthedocs.io/en/latest/_autosummary/grain.experimental.ConcatThenSplitIterDataset.html) that allow to minimize padding or avoid it altogether.

```{code-cell} ipython3
target_length = 70

def process_transformer(sample: dict) -> dict:
  tokenized = tokenizer(sample["review"])["input_ids"][:target_length]
  sample["review"] = np.pad(
      tokenized, pad_width=(0, target_length-len(tokenized)))
  return sample


dataset = (
    grain.MapDataset.source(hf_train)
    .shuffle(seed=42)
    .map(process_date)
    .map(process_transformer)
)
```

Now samples are less human- but more machine-friendly.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo: {}
outputId: aba3bda6-6cc5-4417-9599-7292670e18b3
---
dataset[2]
```

## Complete Pipeline

Time to build our final pipeline! The pipeline doesn't need to be restricted to
`shuffle` and `map`. Grain has a rich API and hands us multiple functionalities
such as: `filter`, `random_map`, `repeat`. Check out
[Grain API](https://google-grain.readthedocs.io/en/latest/grain.html) page to learn more.

On top of the transformer we want to discard reviews that are rated three stars
or less. It's crucial to mention that filtering changes the number of samples in
the following steps so random access is no longer available. To perform
`batching` as the final step we plug `.to_iter_dataset()` converting
`MapDataset` to `IterDataset` - a dataset that gives us an iterator-like
interface.

```{code-cell} ipython3
dataset = (
    grain.MapDataset.source(hf_train)
    .shuffle(seed=42)
    .filter(lambda x: x["star"] > 3)  # filters samples
    .map(process_date)
    .map(process_transformer)
    .map(process_sample_to_np)
    .to_iter_dataset()
    .batch(batch_size=2)  # batches consecutive elements
)
```

With `IterDataset` we can use Python built-ins, `iter` and `next`, to interact
with the dataset.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
executionInfo: {}
outputId: f6836eb9-8878-4e84-a330-0c34b4c7d78d
---
ds_iter = iter(dataset)
next(ds_iter)
```

And that's it! We ended up with a batch with processed date, tokenized review,
and filtered rating.
