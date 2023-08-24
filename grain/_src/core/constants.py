# Copyright 2023 Google LLC
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
"""Shared constants for various Grain APIs."""

# Below are names of meta features used by index_dataset (and pipelines building
# on top of it). These features are generated on the fly and help to track
# progress over the dataset. Users can read these but shouldn't alter them. They
# start with "_" to indicate that they are "private".
# Index into the stream of all records (globally unique). Starts with 0.
INDEX = "_index"
# Key of the record. If DATASET_INDEX is present it's the key in the dataset.
# Starts with 0.
RECORD_KEY = "_record_key"
# Index of the dataset from which to take the record. Only present when mixing.
# Starts with 0.
DATASET_INDEX = "_dataset_index"
# Epoch for the record. When mixing datasets this is the epoch over the dataset,
# not the mixture. Starts with 1.
EPOCH = "_epoch"
# Random seed for stateless random operations. This is unique per record
# and changes every epoch.
SEED = "_seed"
# Serialized record.
RECORD = "_record"

META_FEATURES = frozenset(
    [INDEX, RECORD_KEY, DATASET_INDEX, EPOCH, SEED, RECORD]
)
