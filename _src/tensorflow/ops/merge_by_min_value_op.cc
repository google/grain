/* Copyright 2023 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// See merge_by_min_value.py for a description of the op.
// This file contains the C++ implement as a tf.data op and probably hard to
// read (unless you know tf.data very well).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "third_party/tensorflow/core/data/dataset_utils.h"
#include "third_party/tensorflow/core/data/name_utils.h"
#include "third_party/tensorflow/core/framework/dataset.h"
#include "third_party/tensorflow/core/framework/dataset_options.proto.h"
#include "third_party/tensorflow/core/framework/op_kernel.h"
#include "third_party/tensorflow/core/framework/partial_tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_util.h"
#include "third_party/tensorflow/core/framework/types.h"
#include "third_party/tensorflow/core/kernels/fill_functor.h"
#include "third_party/tensorflow/core/lib/core/errors.h"
#include "third_party/tensorflow/core/lib/gtl/cleanup.h"
#include "third_party/tensorflow/core/platform/blocking_counter.h"
#include "third_party/tensorflow/core/platform/macros.h"
#include "third_party/tensorflow/core/platform/status.h"
#include "third_party/tensorflow/tsl/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {

/*
Structure of this file:
1) Register the op.
2) Define constants and helpers methods.
3) Implement
   - MergeByMinValueDatasetOp
   - MergeByMinValueDatasetOp::Dataset (static shape checking etc.)
   - MergeByMinValueDatasetOp::Dataset::Iterator (actual iterator logic)
4) Register MergeByMinValueDatasetOp as kernel for our op.

This structure is fairly common for TF ops and tf.data ops require the
::Dataset and ::Dataset::Iterator classes.
*/

REGISTER_OP("MergeByMinValueDataset")
    .Input("input_datasets: N * variant")
    .Input("component_index: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .Attr("metadata: string = ''")
    .SetTypeConstructor(full_type::VariadicTensorContainer(TFT_DATASET,
                                                           "output_types"))
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle count_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &count_shape));
      return shape_inference::ScalarShape(c);
      return OkStatus();
    });

constexpr char kDatasetType[] = "MergeByMinValue";
constexpr char kComponentIndex[] = "component_index";
constexpr char kInputImplsEmpty[] = "input_impls_empty";

// Single element in the dataset (each entry is a component - sometimes also
// called a "feature" of the example)
using Element = std::vector<Tensor>;

class MergeByMinValueDatasetOp : public DatasetOpKernel {
 public:
  explicit MergeByMinValueDatasetOp(OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {}

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
};

class MergeByMinValueDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const std::vector<DatasetBase*>& inputs,
          int64_t component_index)
      : DatasetBase(DatasetContext(ctx)),
        inputs_(inputs),
        component_index_(component_index) {
    for (const auto& input : inputs_) {
      input->Ref();
    }
    OP_REQUIRES(ctx, output_dtypes()[component_index] == DT_INT64,
                errors::InvalidArgument(
                    "Component must be a int64 scalar but got dtype ",
                    output_dtypes()[component_index]));
    const auto& shape = output_shapes()[component_index];
    OP_REQUIRES(ctx, shape.IsFullyDefined() && shape.dims() == 0,
                errors::InvalidArgument(
                    "Component must be a int64 scalar but got shape ",
                    output_shapes()[component_index]));
  }

  ~Dataset() override {
    for (const auto& input : inputs_) {
      input->Unref();
    }
  }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return inputs_[0]->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return inputs_[0]->output_shapes();
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
    CardinalityOptions options;
    return CardinalityInternal(options);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    int64_t total = 0;
    for (const auto& input : inputs_) {
      int64_t c = input->Cardinality(options);
      if (c == kInfiniteCardinality) {
        return kInfiniteCardinality;
      }
      if (c == kUnknownCardinality) {
        return kUnknownCardinality;
      }
      total += c;
    }
    return total;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    for (const auto& input : inputs_) {
      inputs->push_back(input);
    }
    return OkStatus();
  }

  Status CheckExternalState() const override {
    for (const auto& input : inputs_) {
      TF_RETURN_IF_ERROR(input->CheckExternalState());
    }
    return OkStatus();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    std::vector<Node*> input_graph_nodes;
    input_graph_nodes.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      Node* input_node;
      TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input, &input_node));
      input_graph_nodes.emplace_back(input_node);
    }
    Node* component_index;
    TF_RETURN_IF_ERROR(b->AddScalar(component_index_, &component_index));
    TF_RETURN_IF_ERROR(
        b->AddDataset(this,
                      /*inputs=*/{std::make_pair(1, component_index)},
                      /*list_inputs=*/{std::make_pair(0, input_graph_nodes)},
                      /*attrs=*/{},
                      /*output=*/output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(mu_);
      input_impls_.resize(dataset()->inputs_.size());
      for (size_t i = 0; i < input_impls_.size(); ++i) {
        TF_RETURN_IF_ERROR(dataset()->inputs_[i]->MakeIterator(
            ctx, this, strings::StrCat(prefix(), "[", i, "]"),
            &input_impls_[i]));
      }
      initialized_heap_ = false;
      return OkStatus();
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      /* We maintain a min heap that contains the component value of the next
      element for each input dataset. If an input dataset is exhausted it is
      no longer in the heap.
      On each call we get the top of the heap which tells us the element to
      return. Afterwards we add a new element to the heap.
      */
      mutex_lock l(mu_);

      // Initialize heap by getting the first element from each input.
      if (!initialized_heap_) {
        initialized_heap_ = true;
        next_elements_.resize(input_impls_.size());
        for (size_t i = 0; i < input_impls_.size(); ++i) {
          bool end;
          TF_RETURN_IF_ERROR(
              input_impls_[i]->GetNext(ctx, &next_elements_[i], &end));
          if (end) continue;
          const uint64_t value = static_cast<uint64_t>(
              next_elements_[i][dataset()->component_index_]
                  .scalar<int64_t>()());
          min_heap_.emplace_back(value, i);
        }
        std::make_heap(min_heap_.begin(), min_heap_.end(), std::greater<>{});
      }

      // If the heap is empty we excausted all inputs.
      if (min_heap_.empty()) {
        *end_of_sequence = true;
        return OkStatus();
      }

      // Pop the top of the heap and set the output element.
      std::pop_heap(min_heap_.begin(), min_heap_.end(), std::greater<>{});
      const size_t top = std::get<1>(min_heap_.back());
      min_heap_.pop_back();
      std::swap(next_elements_[top], *out_tensors);
      *end_of_sequence = false;

      // Try to add another element from the input we just used.
      bool end;
      TF_RETURN_IF_ERROR(
          input_impls_[top]->GetNext(ctx, &next_elements_[top], &end));
      if (!end) {
        const uint64_t value = static_cast<uint64_t>(
            next_elements_[top][dataset()->component_index_]
                .scalar<int64_t>()());
        min_heap_.emplace_back(value, top);
        std::push_heap(min_heap_.begin(), min_heap_.end(), std::greater<>{});
      }
      return OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      // NOTE: Although this dataset may have multiple inputs, it always
      // consumes one element per input to produce an output.
      return model::MakeKnownRatioNode(std::move(args),
                                       /*ratio=*/1);
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name(kInputImplsEmpty),
                              static_cast<int64_t>(input_impls_.empty())));
      for (auto& input_impl : input_impls_) {
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl));
      }
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      int64_t inputs_empty;
      TF_RETURN_IF_ERROR(
          reader->ReadScalar(full_name(kInputImplsEmpty), &inputs_empty));
      if (static_cast<bool>(inputs_empty)) {
        input_impls_.clear();
      } else {
        DCHECK_EQ(input_impls_.size(), dataset()->inputs_.size());
        for (auto& input_impl : input_impls_)
          TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl));
      }
      next_elements_.clear();
      min_heap_.clear();
      initialized_heap_ = false;
      return OkStatus();
    }

   private:
    mutex mu_;
    std::vector<std::unique_ptr<IteratorBase>> input_impls_ TF_GUARDED_BY(mu_);
    bool initialized_heap_;
    // Component value, input dataset index.
    std::vector<std::tuple<int, size_t>> min_heap_;
    // Next value per input dataset. Invalid if there isn't a next element.
    std::vector<Element> next_elements_;
  };  // MergeByMinValueDatasetOp::Dataset::Iterator

  const std::vector<DatasetBase*> inputs_;
  const int64_t component_index_;
};  // MergeByMinValueDatasetOp::Dataset

void MergeByMinValueDatasetOp::MakeDataset(OpKernelContext* ctx,
                                           DatasetBase** output) {
  std::vector<DatasetBase*> inputs;
  for (size_t i = 0; i < ctx->num_inputs() - 1; ++i) {
    DatasetBase* input;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(i), &input));
    inputs.push_back(input);
  }
  int64_t component_index;
  OP_REQUIRES_OK(ctx, ParseScalarArgument<int64_t>(ctx, kComponentIndex,
                                                   &component_index));
  *output = new Dataset(ctx, inputs, component_index);
}

REGISTER_KERNEL_BUILDER(Name("MergeByMinValueDataset").Device(DEVICE_CPU),
                        MergeByMinValueDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
