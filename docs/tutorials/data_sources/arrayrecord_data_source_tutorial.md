---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="1UO5sEL_3t-K" -->
# Reading ArrayRecord Files
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google/grain/blob/main/docs/tutorials/arrayrecord_data_source_tutorial.ipynb)

This tutorial provides an example of how to retrieve records from ArrayRecord
 files using `grain.sources.ArrayRecordDataSource`, also covers how to process
 and transform the data with Grain.

<!-- #endregion -->

<!-- #region id="1p6V-crf3t-K" -->
## Install and Load Dependencies
<!-- #endregion -->

```python id="tzWZLNklr4Iy"
!pip install grain array_record
```

```python id="8NF4E-cCbyjV"
import pickle
import grain
import tensorflow_datasets as tfds
from array_record.python import array_record_module
```

<!-- #region id="cBwdOjDn3t-K" -->
## Write a temp ArrayRecord file
<!-- #endregion -->

```python id="WrCQ-jH53t-K"
# Load a public tensorflow dataset.
test_tfds = tfds.data_source("bool_q", split="train")
```

```python id="_0yBaN7hXmbu"
# Write the dataset into a test array_record file.
example_file_path = "./test.array_record"
writer = array_record_module.ArrayRecordWriter(
    example_file_path, "group_size:1"
)
record_count = 0
for record in test_tfds:
  writer.write(pickle.dumps(record))
  record_count += 1
writer.close()

print(
    f"Number of records written to array_record file {example_file_path} :"
    f" {record_count}"
)
```

```python id="HKJ_49JCXmbu"
# @title Load Data Source
example_array_record_data_source = (grain.sources.ArrayRecordDataSource(
    example_file_path
))
print(f"Number of records: {len(example_array_record_data_source)}")
```

```python id="NVRGllY3Xmbu"
print(example_array_record_data_source[0])
```

<!-- #region id="J2nXJLVUXmbu" -->
## Define Transformation Function
<!-- #endregion -->

```python id="0AS5w9quXmbu"
# Load a pre trained tokenizer
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-cased")
```

```python id="YiS85paBXmbu"
class ParseAndTokenizeText(grain.transforms.Map):
  """This function takes a serialized dict (as bytes), decodes it,

  applies a tokenizer to a specified feature within the dict,
  and returns the first 10 tokens from results.
  """

  def __init__(self, tokenizer, feature_name):
    self._tokenizer = tokenizer
    self._feature_name = feature_name

  def map(self, element: bytes) -> [str]:
    parsed_element = pickle.loads(element)
    # only pick the first 10 token IDs from the tokenized text for testing
    return self._tokenizer.encode(
        parsed_element[self._feature_name].decode('utf-8')
    ).tokens[:10]
```

<!-- #region id="fLqi3i7O3t-K" -->
## Load and process data via the Dataset API
<!-- #endregion -->

```python id="RPIy05gGUBzI"
# Example using Grain's MapDataset with ArrayRecord file source.
example_datasets = (
    grain.MapDataset.source(example_array_record_data_source)
    .shuffle(seed=42)
    .map(ParseAndTokenizeText(tokenizer, "question"))
    .batch(batch_size=10)
)
```

```python id="xqJSeQ9hdAmF"
# Output a record at a random index
print(example_datasets[100])
```
