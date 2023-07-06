"""Provides a methods to run functions in parallel using a thread pool."""
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar

from concurrent import futures

T = TypeVar("T")


def run_in_parallel(
    function: Callable[..., T],
    list_of_kwargs_to_function: Sequence[Mapping[str, Any]],
    num_workers: int,
    thread_name_prefix: str = "parallel_",
) -> list[T]:
  """Run a function on a list of kwargs in parallel with ThreadPoolExecutor.

  Works best when there is IO boundedness, not when there is CPU boundedness, as
  the threads used are bound by the GIL.

  Propagates first exception to the calling thread. If cancel_futures=True,
  then stop as many of the ongoing work units as possible.

  Example usage:
    def io_bound_function(p):
      get_contents_from_cns(p)

    run_in_parallel(
        function=io_bound_function,
        list_of_kwargs_to_function=[{"p": p} for p in long_list_of_paths],
        num_workers=3)

  Args:
    function: a function.
    list_of_kwargs_to_function: A list of dicts mapping from string to argument
      value. These will be passed into `function` as kwargs.
    num_workers: int.
    thread_name_prefix: The thread name prefix string. Processes are run in
      threads, and each thread is named. This parameter allows the user to
      control the prefix for that thread name.

  Returns:
    list of return values from function, in the same order as the arguments in
    list_of_kwargs_to_function.
  """
  if num_workers < 1:
    raise ValueError(
        "Number of workers must be greater than 0. Was {}".format(num_workers)
    )

  thread_name = thread_name_prefix + getattr(function, "__name__", "unknown")
  with futures.ThreadPoolExecutor(
      num_workers, thread_name_prefix=thread_name
  ) as executor:
    fs = []

    for kwargs in list_of_kwargs_to_function:
      f = executor.submit(function, **kwargs)
      fs.append(f)

    futures_as_completed = futures.as_completed(fs)

    for completed in futures_as_completed:
      if completed.exception():
        # Cancel all remaining futures, if possible.
        for remaining_future in fs:
          remaining_future.cancel()

        # Propagate exception to main thread.
        raise completed.exception()

  return [f.result() for f in fs]
