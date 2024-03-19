# Samplers



https://github.com/google/grain/tree/main/docs/samplers.md

Samplers in PyGrain are responsible for determining the order in which records
are processed. This allows PyGrain to implement global transformations (e.g.
global shuffling, sharding, repeating for multiple epochs) before reading any
records.



Samplers need to implement the following iterator protocol:

```python
class Sampler(Protocol):

  def __iter__(self):
    ...

  def __next__(self) -> record.RecordMetadata:
    ...

@dataclasses.dataclass
class RecordMetadata:
  """RecordMetadata contains metadata about individual records.

  RecordMetadata objects are emitted by the sampler to refer to which record to
  read next (record_key), what its index is (for keeping progress and
  checkpointing) as well as having an optional rng for stateless random
  transformations. In addition, they are also used to keep information about
  records as they flow through the pipeline from one operation to the other.
  """
  index: int
  record_key: Optional[int] = None
  rng: Optional[np.random.Generator] = None
```

## Index Sampler
This is our recommended Sampler. It supports:

* Sharding across multiple machines (`shard_options` parameter).
* Global shuffle of the data (`shuffle` parameter).
* Repeating records for multiple epochs (`num_epochs` parameter). Note that the
shuffle order changes across epochs. Behind the scenes, this relies on
[tf.random_index_shuffle](https://www.tensorflow.org/api_docs/python/tf/random_index_shuffle).
* Stateless random operations. Each `RecordMetadata` object emitted by the 
`IndexSampler` contains an RNG uniquely seeded on a per-record basis. This
RNG can be used for random augmentations while not relying on a global state.

```python
index_sampler = pygrain.IndexSampler(
  num_records=5,
  num_epochs=2,
  shard_options=pygrain.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True),
  shuffle=True,
  seed=0)
for record_metadata in index_sampler:
  print(record_metadata)

# Output
# RecordMetadata(index=0, record_key=0, rng=Generator(Philox) at 0x7FB09947AF80)
# RecordMetadata(index=1, record_key=4, rng=Generator(Philox) at 0x7FB0994789E0)
# RecordMetadata(index=2, record_key=2, rng=Generator(Philox) at 0x7FB099478740)
# RecordMetadata(index=3, record_key=3, rng=Generator(Philox) at 0x7FB0994789E0)
# RecordMetadata(index=4, record_key=1, rng=Generator(Philox) at 0x7FB099478740)
# RecordMetadata(index=5, record_key=1, rng=Generator(Philox) at 0x7FB0994789E0)
# RecordMetadata(index=6, record_key=0, rng=Generator(Philox) at 0x7FB099478740)
# RecordMetadata(index=7, record_key=3, rng=Generator(Philox) at 0x7FB0994789E0)
# RecordMetadata(index=8, record_key=4, rng=Generator(Philox) at 0x7FB099478740)
# RecordMetadata(index=9, record_key=2, rng=Generator(Philox) at 0x7FB0994789E0)
```

## Implement your Own Sampler
PyGrain can accommodate custom user-defined samplers. Users implementing their
own sampler should ensure it:

* implements the aforementioned interface.
* is adequately performant. Since PyGrain's
`DataLoader` iterates sequentially through the sampler to distribute indices to
child processes, a slow sampler will become a bottleneck and reduce end-to-end
pipeline performance. As a reference, we recommend sampler iteration performance
of at approx. 50,000 elements / sec for most use cases.
