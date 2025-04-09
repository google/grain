"""API to test checkpointing."""

import itertools
from typing import Any

from grain._src.core import tree_lib
import numpy as np


def assert_equal_output_after_checkpoint(
    ds: Any,
):
  """Tests restoring an iterator to various checkpointed states.

  Args:
    ds: The dataset to test.  It is recommended to use a small dataset,
      potentially created using `grain.python.experimental.LimitIterDataset`, to
      restrict the number of steps being tested. The underlying dataset iterator
      must implement `get_state` and `set_state` for checkpointing.
  """

  iterator = ds.__iter__()
  checkpoints = []
  expected_values = []
  state_spec = None
  for i in itertools.count():
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
    try:
      value = next(iterator)
    except StopIteration:
      break
    checkpoints.append(current_state)
    expected_values.append(value)

  assert expected_values, "Dataset did not produce any elements."

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
    new_values = list(new_iterator)
    np.testing.assert_equal(
        new_values,
        expected_values[i:],
        f"Restored values mismatch at step {i} for state {state}."
        f" \nExpected: {expected_values[i:]}, \nActual: {new_values}",
    )
