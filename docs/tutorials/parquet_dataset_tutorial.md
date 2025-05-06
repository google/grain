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

<!-- #region id="DQk1knFmoaw_" -->
# Reading Apache Parquet

This tutorial provides an example of how to read data from [Apache Parquet](https://parquet.apache.org/) file, also covers how to process and transform the data with Grain.

<!-- #endregion -->

<!-- #region id="ZW0hSbMYtn0s" -->
## Generate a test Parquet file on local
<!-- #endregion -->

```python id="dFnwN6NNw0Oe"
import pyarrow as pa
import pyarrow.parquet as pq

# Generate a sample PyArrow table containing email subjects and bodies.
table = pa.table({
    'email_subject': [
        "Meeting Reminder: Project X Update",
        "Important Announcement Regarding Company Policy",
        "FWD: Quick Question",
        "Your Order Confirmation #12345",
        "Invitation to Team Building Activity"
    ],
    'email_body': [
        "Hi team,\n\nJust a reminder about our Project X update meeting tomorrow at 10 AM PST. Please come prepared to discuss your progress and any roadblocks.\n\nSee you there,\n[Your Name]",
        "Dear employees,\n\nPlease be advised of a new company policy regarding remote work, effective May 1st, 2025. You can find the full details on the company intranet.\n\nRegards,\nManagement",
        "Hi [Name],\n\nForwarding you this email as you might have the answer to this quick question:\n\n[Original Email Content]",
        "Dear [Customer Name],\n\nThank you for your recent order! This email confirms your order #12345. You can view the details and track its shipment here: [Link]\n\nSincerely,\nThe [Company Name] Team",
        "Hello everyone,\n\nYou're invited to participate in our upcoming team building activity on Friday, April 28th. It will be a fun afternoon of [Activity]. Please RSVP by Wednesday.\n\nBest,\n[Organizer Name]"
    ]
})

# Write this table to a parquet file.
writer = pq.ParquetWriter('emails.parquet', table.schema)
writer.write_table(table)
writer.close()

```

<!-- #region id="-fvKIjtRD7v9" -->
## Load Dataset
<!-- #endregion -->

```python id="fNIApmKT34ac"
# Install Grain
!pip install grain
```

```python id="nb5yPasvDwaj"
import grain
import pprint
```

```python id="62_D74LkdrQn"
ds = grain.experimental.ParquetIterDataset('./emails.parquet')
```

```python id="DlhbJX5zdrQo"
list(ds)[0]
```

<!-- #region id="BAXS0bgKdrQo" -->
## Transform Dataset
<!-- #endregion -->

```python id="1qevksHMdrQo"
# Load a pre trained tokenizer.
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-cased")
```

```python id="PNPv8LEGdrQo"
class TokenizeText(grain.transforms.Map):
  """Tokenizes the text values within each element using a provided tokenizer."""
  def __init__(self, tokenizer):
    self._tokenizer = tokenizer

  def map(self, element):
    return [self._tokenizer.encode(item).tokens for item in element.values()]
```

```python id="O5vnwj7cek30"
# Tokenize the data using the provided tokenizer.
ds = ds.map(TokenizeText(tokenizer))
```

```python id="46ZVbfmtek30"
# Create an iterator object of the dataset.
ds_iter = iter(ds)
# Print the first element in the dataset.
pprint.pprint(next(ds_iter))
```
