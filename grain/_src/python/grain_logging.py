"""A library for adding a custom identifier to Python log messages.

It allows adding a prefix identifying the process generating the log statement.
The main purpose is to make logs more readable when using multiprocessing.
"""

import logging
from absl import logging as absl_logging


# Adds a prefix containing the `identifier` to all new Python log messages.
def set_process_identifier_prefix(identifier: str) -> None:
  log_formatter = logging.Formatter(f'[{identifier}] %(message)s')
  absl_logging.get_absl_handler().setFormatter(log_formatter)
