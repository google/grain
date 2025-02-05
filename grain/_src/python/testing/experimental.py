"""API to test checkpointing."""

from grain._src.core import tree_lib
from grain._src.python.dataset import dataset
import numpy as np


# TODO: Introduce `LimitIterDataset(ds, max_steps)` and modify this
# test to use that.
def assert_equal_output_after_checkpoint(
    ds: dataset.IterDataset,
    max_steps: int = 5,
):
  """Tests restoring an iterator to various checkpointed states.."""

  iterator = ds.__iter__()
  checkpoints = []
  values = []
  state_spec = None
  for i in range(max_steps):
    current_state = iterator.get_state()
    if state_spec is None:
      state_spec = tree_lib.spec_like(current_state)
    else:
      np.testing.assert_equal(
          state_spec,
          tree_lib.spec_like(current_state),
          f"State spec does not match the original state spec at step {i}."
          f" Expected: {state_spec},"
          f" Actual: {tree_lib.spec_like(current_state)}",
      )
    checkpoints.append(current_state)
    values.append(next(iterator))

  # Restore the iterator at every state, and compare the values.
  for i, state in enumerate(checkpoints):
    new_iterator = ds.__iter__()
    new_iterator.set_state(state)
    np.testing.assert_equal(
        new_iterator.get_state(),
        state,
        f"Restored state does not match the original state at step {i}."
        f" Expected: {state}, Actual: {new_iterator.get_state()}",
    )

    # Test the values at the current state.
    new_values = [next(new_iterator) for _ in range(min(max_steps - i, 3))]
    np.testing.assert_equal(
        new_values,
        values[i : i + 3],
        f"Restored values mismatch at step {i} for state {state}."
        f" \nExpected: {values[i:i+3]}, \nActual: {new_values}",
    )
