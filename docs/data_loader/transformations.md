# Transformations



.md



Grain Transforms interface denotes transformations which are applied to data. In
the case of local transformations (such as map, random map, filter), the
transforms receive an element on which custom changes are applied. For global
transformations (such as batching), one must provide the batch size.

The Grain core transforms interface code is
[here](https://github.com/google/grain/tree/main/grain/_src/core/transforms.py).

## MapTransform

`MapTransform` is for 1:1 transformations of elements. Elements can be of any
type, it is the user's responsibility to use the transformation such that the
inputs it receives correspond to the signature.

Example of transformation which implements `MapTransform` (for elements of type
`int`):

```python
class PlusOne(transforms.MapTransform):

  def map(self, x: int) -> int:
    return x + 1
```

## MapWithIndexTransform

`MapWithIndexTransform` is similar to `MapTransform` in being a 1:1
transformations of elements, but also takes in the index/position of the element
as the first argument. This is useful for pairing elements with an index key or
even keeping it as metadata alongside the actual data.

Example of transformation which implements `MapWithIndexTransform` (for elements
of type `int`):

```python
class PlusOneWithIndexKey(transforms.MapWithIndexTransform):

  def map_with_index(self, i: int, x: int) -> tuple[int, int]:
    return (x + 1, i)
```

## RandomMapTransform

`RandomMapTransform` is for 1:1 random transformations of elements. The
interface requires a `np.random.Generator` as parameter to the `random_map`
function.

Example of a `RandomMapTransform`:

```python
class PlusRandom(transforms.RandomMapTransform):

  def random_map(self, x: int, rng: np.random.Generator) -> int:
    return x + rng.integers(100_000)
```

## FlatMapTransform

`FlatMapTransform` is for splitting operations of individual elements. The
`max_fan_out` is the maximum number of splits that an element can generate.
Please consult the code for detailed info.

Example of a `FlatMapTransform`:

```python
class FlatMapTransformExample(transforms.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element: int):
    for _ in range(self.max_fan_out):
      yield element
```

## FilterTransform

`FilterTransform` is for applying filtering to individual elements. Elements for
which the filter function returns False will be removed.

Example of a `FilterTransform` that removes all even elements:

```python
class RemoveEvenElements(FilterTransform):

  def filter(self, element: int) -> bool:
    return element % 2
```

## Batch

To apply the `Batch` transform, pass `grain.Batch(batch_size=batch_size,
drop_remainder=drop_remainder)`.

Note: The batch size used when passing `Batch` transform will be the global
batch size if it is done before sharding and the *per host* batch size if it is
after. Typically usage with `IndexSampler` is after sharding.
