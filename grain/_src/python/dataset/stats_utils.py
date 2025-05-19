# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Util functions for stats.py."""

import functools
import math
from typing import Any

from absl import logging
from grain._src.core import tree_lib
from grain._src.python import shared_memory_array
from grain.proto import execution_summary_pb2
import numpy as np

_SOURCE_NODE_NAME = "SourceMapDataset"


def is_source_node(node: execution_summary_pb2.ExecutionSummary.Node) -> bool:
  """Returns True if the node is a source node."""
  return _SOURCE_NODE_NAME in node.name


def analyze_summary(
    summary: execution_summary_pb2.ExecutionSummary,
) -> list[str]:
  """Analyzes the sorted execution summary and returns warning messages."""
  warnings = []
  num_nodes = len(summary.nodes)
  # The execution summary is sorted by wait time ratio by the time we
  # analyze it.
  top_node = summary.nodes[num_nodes - 1]
  if is_source_node(top_node):
    msg = (
        "WARNING: Your source is likely the bottleneck. Please ensure if you"
        " have enough spindle quota or if your data is co-located with the"
        " computation. "
    )
    warnings.append(msg)
  return warnings


def sort_nodes_by_wait_time_ratio(
    execution_summary: execution_summary_pb2.ExecutionSummary,
) -> execution_summary_pb2.ExecutionSummary:
  """Sorts the nodes in the summary by their wait time ratio."""
  nodes = list(execution_summary.nodes.values())
  nodes.sort(key=lambda x: x.wait_time_ratio)
  execution_summary.ClearField("nodes")
  for node_id, node in enumerate(nodes):
    execution_summary.nodes[node_id].CopyFrom(node)
  return execution_summary


def format_ratio_as_percent(value: float) -> str:
  return f"{value*100:.2f}%"


def pretty_format_ns(value: int) -> str:
  """Pretty formats a time value in nanoseconds to human readable value."""
  if value < 1000:
    return f"{value}ns"
  elif value < 1000_000:
    return f"{value/1000:.2f}us"
  elif value < 1_000_000_000:
    return f"{value/1000_000:.2f}ms"
  else:
    return f"{value/1000_000_000:.2f}s"


def get_avg_processing_time_ns(
    node: execution_summary_pb2.ExecutionSummary.Node,
) -> int:
  """Returns the average processing time in nanoseconds for the given node."""
  if node.num_produced_elements == 0:
    return 0
  return int(node.total_processing_time_ns / node.num_produced_elements)


def pretty_format_bytes(bytes_value: int) -> str:
  """Returns a pretty formatted string for bytes."""
  # pylint: disable=bad-whitespace
  if bytes_value < 1024:
    return f"{bytes_value} bytes"
  elif bytes_value < 1024 * 1024:
    return f"{bytes_value / 1024:.2f} KiB"
  elif bytes_value < 1024 * 1024 * 1024:
    return f"{bytes_value / 1024 / 1024:.2f} MiB"
  elif bytes_value < 1024 * 1024 * 1024 * 1024:
    return f"{bytes_value / 1024 / 1024 / 1024:.2f} GiB"
  else:
    return f"{bytes_value / 1024 / 1024 / 1024 / 1024:.2f} TiB"
  # pylint: enable=bad-whitespace


def merge_execution_summaries(
    from_s: execution_summary_pb2.ExecutionSummary,
    to_s: execution_summary_pb2.ExecutionSummary,
) -> execution_summary_pb2.ExecutionSummary:
  """Merges the `from_s` execution summary into the `to_s` execution summary."""
  # We cannot use MergeFrom here because singular fields like
  # `max_processing_time_ns` will be overriden.
  for node_id in from_s.nodes:
    to_node = to_s.nodes[node_id]
    from_node = from_s.nodes[node_id]
    to_node.id = from_s.nodes[node_id].id
    to_node.name = from_node.name
    to_node.output_spec = from_node.output_spec
    to_node.is_output = from_node.is_output
    to_node.ClearField("inputs")
    to_node.inputs.extend(from_node.inputs)
    to_node.min_processing_time_ns = min(
        to_node.min_processing_time_ns,
        from_node.min_processing_time_ns,
    )
    to_node.max_processing_time_ns = max(
        to_node.max_processing_time_ns,
        from_node.max_processing_time_ns,
    )
    to_node.total_processing_time_ns += from_node.total_processing_time_ns
    to_node.num_produced_elements += from_node.num_produced_elements
    to_node.bytes_produced += from_node.bytes_produced
    to_node.bytes_consumed += from_node.bytes_consumed
  return to_s


def update_node_ids_with_offset(
    summary: execution_summary_pb2.ExecutionSummary,
    num_nodes: int,
) -> None:
  """Updates node IDs in a summary by adding `num_nodes` offset."""
  for node in summary.nodes.values():
    node.id += num_nodes
    current_input_ids = node.inputs
    node.ClearField("inputs")
    node.inputs.extend([input_id + num_nodes for input_id in current_input_ids])


def get_output_nodes(
    summary: execution_summary_pb2.ExecutionSummary,
) -> list[execution_summary_pb2.ExecutionSummary.Node]:
  """Returns the output node ID in the execution summary."""
  output_nodes = []
  for node in summary.nodes.values():
    if getattr(node, "is_output"):
      output_nodes.append(node)
  return output_nodes


def get_complete_summary(
    main_summary: execution_summary_pb2.ExecutionSummary,
    workers_summary: execution_summary_pb2.ExecutionSummary,
    num_nodes_in_main,
) -> execution_summary_pb2.ExecutionSummary:
  """Updates the nodes in the main summary with workers summary nodes."""
  # The pipeline's output node within the workers becomes the input for the
  # root node in the main process. Therefore, the node IDs (and thus input
  # IDs) in the worker summary should be updated before merging it with the
  # main process's summary.
  update_node_ids_with_offset(workers_summary, num_nodes_in_main)
  # The output nodes in workers summary are the inputs to the main process.
  output_node_ids = [node.id for node in get_output_nodes(workers_summary)]

  total_bytes_produced_by_workers = 0
  for node in get_output_nodes(workers_summary):
    node.is_output = False
    total_bytes_produced_by_workers += node.bytes_produced

  root_id = num_nodes_in_main - 1
  main_summary.nodes[root_id].inputs.extend(output_node_ids)
  # `bytes_consumed`` for the root node in the main process is the sum of
  # `bytes_produced` by the output node in the workers summary.
  main_summary.nodes[root_id].bytes_consumed = total_bytes_produced_by_workers

  # Merge the workers summary into the main summary.
  for node in workers_summary.nodes.values():
    main_summary.nodes[node.id].CopyFrom(node)

  return main_summary


def calculate_allocated_bytes(element: Any) -> int:
  """Calculates the allocated bytes of the element.

  This method estimates the memory footprint of common data types within Grain
  data pipelines.

  Args:
    element: The element for which the size is to be calculated.

  Returns:
    Estimated allocated bytes in memory for the element, or 0 for unsupported
    types.
  """
  if isinstance(element, np.ndarray):
    return element.nbytes
  elif isinstance(element, shared_memory_array.SharedMemoryArrayMetadata):
    return math.prod(element.shape) * np.dtype(element.dtype).itemsize
  elif isinstance(element, str) or isinstance(element, bytes):
    return len(element)
  else:
    logging.log_first_n(
        logging.WARNING,
        "Unsupported type for estimating memory usage: %s",
        1,
        type(element),
    )
    return 0


def _record_leaf_size(element: Any, memory_buffer: list[int]) -> Any:
  """Calculates the size of leaf element and appends it to the memory buffer."""
  size = calculate_allocated_bytes(element)
  memory_buffer.append(size)
  return element


def record_size(element: Any, memory_buffer: list[int]) -> Any:
  """Records the size of the element in the memory buffer."""
  return tree_lib.map_structure(
      functools.partial(_record_leaf_size, memory_buffer=memory_buffer),
      element,
  )
