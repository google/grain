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

from grain.proto import execution_summary_pb2


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
  for node in get_output_nodes(workers_summary):
    node.is_output = False
  root_id = num_nodes_in_main - 1
  main_summary.nodes[root_id].inputs.extend(output_node_ids)

  # Merge the workers summary into the main summary.
  for node in workers_summary.nodes.values():
    main_summary.nodes[node.id].CopyFrom(node)

  return main_summary
