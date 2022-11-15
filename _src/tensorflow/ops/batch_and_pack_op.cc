/* Copyright 2022 Google LLC. All Rights Reserved.

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
// See batch_and_pack.py for a description of the op.
// This file contains the C++ implement as a tf.data op and probably hard to
// read (unless you know tf.data very well).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "third_party/tensorflow/core/data/dataset_utils.h"
#include "third_party/tensorflow/core/data/name_utils.h"
#include "third_party/tensorflow/core/framework/dataset.h"
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
#include "third_party/tensorflow/core/util/batch_util.h"

namespace tensorflow {
namespace data {
namespace {

/*
Structure of this file:
1) Register the op.
2) Define constants and helpers methods.
3) Define PackedBatch which handles most of the actual packing logic.
4) Implement
   - BatchAndPackDatasetOp
   - BatchAndPackDatasetOp::Dataset (static shape checking etc.)
   - BatchAndPackDatasetOp::Dataset::Iterator (actual iterator logic)
5) Register BatchAndPackDatasetOp as kernel for our op.

This structure is fairly common for TF ops and tf.data ops require the
::Dataset and ::Dataset::Iterator classes.
*/

REGISTER_OP("BatchAndPackDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("sequence_lengths: N * int64")
    .Output("handle: variant")
    .Attr("parallel_copy: bool = false")
    .Attr("Toutput_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .Attr("metadata: string = ''")
    .SetTypeConstructor(full_type::VariadicTensorContainer(TFT_DATASET,
                                                           "Toutput_types"))
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

constexpr char kDatasetType[] = "BatchAndPack";
constexpr char kBatchSize[] = "batch_size";
constexpr char kSequenceLengths[] = "sequence_lengths";
constexpr char kParallelCopy[] = "parallel_copy";
constexpr char kToutputTypes[] = "Toutput_types";
constexpr char kNumPaddedShapes[] = "N";
constexpr char kExhausted[] = "exhausted";

using CPUDevice = Eigen::ThreadPoolDevice;
// Single element in the dataset (each entry is a component - sometimes also
// called a "feature" of the example)
using Element = std::vector<Tensor>;

// Returns a-b for vectors a and b.
template <typename T>
std::vector<T> element_wise_minus(const std::vector<T>& a,
                                  const std::vector<T>& b) {
  assert(a.size() == b.size());
  std::vector<T> result;
  result.reserve(a.size());
  std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result),
                 std::minus<T>());
  return result;
}

// Copy an element into a slice of the parent tensor. The slices is given by
// the starting positions for the first 2 dimensions and the size of the
// element.
// end = index_dim1 + element.shape[0]
// parent[index_dim0, index_dim1:end, ...] = element
//
// This is the fully templated version. You probably want to use
// CopyElementToLargerSlice.
template <typename T, int NDIMS>
Status CopyElementToLargerSliceWithTypeAndRank(const Tensor& element,
                                               Tensor* parent, int index_dim0,
                                               int index_dim1) {
  if (element.NumElements() == 0) {
    return OkStatus();
  }

  // Treat scalar as vector of size [1].
  if (NDIMS == 0) {
    Tensor scalar_as_vector;
    if (!scalar_as_vector.CopyFrom(element, {1})) {
      return errors::InvalidArgument(
          "Could not treat scalar as vector. Please file a bug.");
    }
    return CopyElementToLargerSliceWithTypeAndRank<T, 1>(
        scalar_as_vector, parent, index_dim0, index_dim1);
  }

  auto element_t = element.tensor<T, NDIMS>();
  auto parent_t = parent->tensor<T, NDIMS + 1>();
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_indices;
  slice_indices[0] = index_dim0;
  slice_indices[1] = index_dim1;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS + 1> slice_size;
  slice_size[0] = 1;
  for (size_t i = 1; i < slice_size.size(); ++i) {
    slice_size[i] = element_t.dimension(i - 1);
  }
  parent_t.slice(slice_indices, slice_size) = element_t.reshape(slice_size);
  return OkStatus();
}

template <int NDIMS>
Status CopyElementToLargerSliceWithRank(const Tensor& element, Tensor* parent,
                                        int index_dim0, int index_dim1) {
#define HANDLE_TYPE(T)                                        \
  case DataTypeToEnum<T>::value: {                            \
    return CopyElementToLargerSliceWithTypeAndRank<T, NDIMS>( \
        element, parent, index_dim0, index_dim1);             \
  }

  switch (element.dtype()) {
    TF_CALL_DATASET_TYPES(HANDLE_TYPE);
#undef HANDLE_TYPE
    default:
      return errors::Unimplemented(
          "HandleElementToLargerSliceWithRank Unhandled data type: ",
          element.dtype());
  }
}

// See CopyElementToLargerSliceWithTypeAndRank.
Status CopyElementToLargerSlice(const Tensor& element, Tensor* parent,
                                int index_dim0, int index_dim1) {
  // Element must be a scalar or fit into the parent slice.
  if (element.dims() > 0 && parent->dims() != element.dims() + 1) {
    return errors::Internal(
        "Mismatched ranks.  Element's rank is: ", element.dims(),
        " but element is meant to be a slice in output Tensor having rank: ",
        parent->dims(), " (should be: ", element.dims() + 1, ")");
  }

#define HANDLE_DIMS(NDIMS)                                      \
  case NDIMS: {                                                 \
    TF_RETURN_IF_ERROR(CopyElementToLargerSliceWithRank<NDIMS>( \
        element, parent, index_dim0, index_dim1));              \
    return OkStatus();                                          \
  }

  switch (element.dims()) {
    HANDLE_DIMS(0);  // Scalar.
    HANDLE_DIMS(1);
    HANDLE_DIMS(2);
    HANDLE_DIMS(3);
    HANDLE_DIMS(4);
    HANDLE_DIMS(5);
#undef HANDLE_DIMS
    default:
      return errors::Unimplemented("CopyElementToLargerSlice Unhandled rank: ",
                                   element.dims());
  }
}

// Aggregates elements for a packed batch.
// The packed batch has `batch_size` rows and each row can fit multiple
// elements.
class PackedBatch {
 public:
  PackedBatch(const int batch_size,
              const std::vector<int64_t>& sequence_lengths)
      : sequence_lengths_(sequence_lengths) {
    elements_.reserve(batch_size);
    row_per_element_.reserve(batch_size);
    column_per_element_.reserve(batch_size);
    free_cells_.reserve(batch_size);
    for (int row = 0; row < batch_size; ++row) {
      free_cells_.push_back(sequence_lengths);
    }
  }

  size_t NumRows() const { return free_cells_.size(); }

  int64_t MaxRow() const {
    if (row_per_element_.empty()) return -1;
    return *std::max_element(std::begin(row_per_element_),
                             std::end(row_per_element_));
  }

  // Number of components of the input elements. The output elements should
  // have 3 times the components (values, segment IDs, positions).
  size_t NumComponents() const { return free_cells_[0].size(); }

  size_t NumElements() const { return elements_.size(); }

  bool Empty() const { return elements_.empty(); }

  bool TryAddElement(Element& element) {
    std::vector<int64_t> required_cells;
    for (const auto& component : element) {
      required_cells.push_back(component.dims() == 0 ? 1
                                                     : component.dim_size(0));
    }
    for (int row = 0; row < NumRows(); ++row) {
      bool fits = true;
      for (int component = 0; component < NumComponents(); ++component) {
        if (free_cells_[row][component] < required_cells[component]) {
          fits = false;
          break;
        }
      }
      if (fits) {
        elements_.push_back(std::move(element));
        row_per_element_.push_back(row);
        column_per_element_.push_back(
            element_wise_minus(sequence_lengths_, free_cells_[row]));
        for (int component = 0; component < NumComponents(); ++component) {
          free_cells_[row][component] -= required_cells[component];
        }
        return true;
      }
    }
    return false;
  }

  Status InsertValues(Tensor* batch_t, int element_index, int component) {
    TF_RETURN_IF_ERROR(CopyElementToLargerSlice(
        elements_[element_index][component], batch_t,
        row_per_element_[element_index],
        column_per_element_[element_index][component]));
    return OkStatus();
  }

  Status AddSegmentIds(Tensor* batch_t, int element_index, int component) {
    if (batch_t->dims() != 2 || batch_t->dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "Segment IDs must be a 2 dimensional tensor of type int64 but got " +
          batch_t->DebugString());
    }
    if (batch_t->dim_size(0) != NumRows()) {
      return errors::InvalidArgument(
          "Expected first dimension of segment IDs to have size ", NumRows(),
          " but got ", batch_t->DebugString());
    }
    const int64_t row = row_per_element_[element_index];
    int64_t segmentation = 1;
    for (int64_t i = 0; i < element_index; ++i) {
      if (row_per_element_[i] == row_per_element_[element_index]) {
        segmentation++;
      }
    }
    const int64_t offset = column_per_element_[element_index][component];
    const int seq_len = (elements_[element_index][component].dims() == 0
                             ? 1
                             : elements_[element_index][component].dim_size(0));
    for (int64_t p = 0; p < seq_len; ++p) {
      batch_t->matrix<int64_t>()(row, p + offset) = segmentation;
    }
    return OkStatus();
  }

  Status AddPositions(Tensor* batch_t, int element_index, int component) {
    if (batch_t->dims() != 2 || batch_t->dtype() != DT_INT64) {
      return errors::InvalidArgument(
          "Positions must be a 2 dimensional tensor of type int64 but got " +
          batch_t->DebugString());
    }
    if (batch_t->dim_size(0) != NumRows()) {
      return errors::InvalidArgument(
          "Expected first dimension of postions to have size ", NumRows(),
          " but got ", batch_t->DebugString());
    }
    const int64_t row = row_per_element_[element_index];
    const int64_t offset = column_per_element_[element_index][component];
    const int seq_len = (elements_[element_index][component].dims() == 0
                             ? 1
                             : elements_[element_index][component].dim_size(0));
    for (int64_t p = 0; p < seq_len; ++p) {
      batch_t->matrix<int64_t>()(row, p + offset) = p;
    }
    return OkStatus();
  }

 private:
  const std::vector<int64_t>& sequence_lengths_;
  std::vector<Element> elements_;
  // row in which each element goes.
  std::vector<int64_t> row_per_element_;
  std::vector<std::vector<int64_t>> column_per_element_;  // For each component.
  // free cells per [row][component].
  std::vector<std::vector<int64_t>> free_cells_;
};

class BatchAndPackDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit BatchAndPackDatasetOp(OpKernelConstruction* ctx)
      : UnaryDatasetOpKernel(ctx) {
    if (ctx->HasAttr(kParallelCopy)) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr(kParallelCopy, &parallel_copy_));
    }
  }

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  bool parallel_copy_ = false;
};

class BatchAndPackDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, int64_t batch_size, bool parallel_copy,
          std::vector<int64_t> sequence_lengths, const DatasetBase* input)
      : DatasetBase(DatasetContext(ctx)),
        batch_size_(batch_size),
        parallel_copy_(parallel_copy),
        sequence_lengths_(std::move(sequence_lengths)),
        input_(input) {
    input_->Ref();

    // For each input we output the values, segment IDs and positions.
    output_shapes_.reserve(3 * input_->output_shapes().size());
    output_dtypes_.reserve(3 * input_->output_dtypes().size());
    for (size_t input_component = 0;
         input_component < input_->output_shapes().size(); ++input_component) {
      const auto& input_shape = input_->output_shapes()[input_component];
      // First 2 dimensions are always (batch size and sequence length).
      PartialTensorShape output_shape(
          {batch_size_, sequence_lengths_[input_component]});
      if (input_shape.dims() > 0) {
        OP_REQUIRES(
            ctx, input_shape.dim_size(0) <= sequence_lengths_[input_component],
            errors::InvalidArgument(
                "First dimension (sequence dimension) for component ",
                input_component, " has size ", input_shape.dim_size(0),
                " but is not allowed to be longer than the provide sequence "
                "length (",
                sequence_lengths_[input_component], ")."));
      }
      for (int dim = 1; dim < input_shape.dims(); ++dim) {
        OP_REQUIRES(ctx, input_shape.dim_size(dim) != -1,
                    errors::InvalidArgument(
                        "Only the first dimension (sequence dimension) can "
                        "have dynamic shape. All other dimensions but be "
                        "statically known. Got unknown size of dimension ",
                        dim, " in component ", input_component, "."));
        output_shape.AddDim(input_shape.dim_size(dim));
      }
      output_shapes_.push_back(output_shape);
      output_dtypes_.push_back(input_->output_dtypes()[input_component]);

      // Segment IDs.
      output_shapes_.push_back(
          TensorShape({batch_size_, sequence_lengths_[input_component]}));
      output_dtypes_.push_back(DT_INT64);
      // Positions.
      output_shapes_.push_back(
          TensorShape({batch_size_, sequence_lengths_[input_component]}));
      output_dtypes_.push_back(DT_INT64);
    }
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    name_utils::IteratorPrefixParams params;
    return std::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix, params)});
  }

  const DataTypeVector& output_dtypes() const override {
    return output_dtypes_;
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  string DebugString() const override {
    name_utils::DatasetDebugStringParams params;
    params.set_args(batch_size_);
    return name_utils::DatasetDebugString(kDatasetType, params);
  }

  int64_t CardinalityInternal() const override {
    int64_t n = input_->Cardinality();
    // Infinite datasets stay infinite.
    if (n == kInfiniteCardinality) {
      return n;
    }
    // Everything else gets unknown cardinality since we cannot know how much
    // elements will pack into each batch.
    return kUnknownCardinality;
  }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return OkStatus();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* batch_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(batch_size_, &batch_size));

    std::vector<Node*> sequence_lengths;
    sequence_lengths.reserve(sequence_lengths_.size());
    for (int i = 0; i < sequence_lengths_.size(); ++i) {
      Node* node;
      Tensor t(sequence_lengths_[i]);
      TF_RETURN_IF_ERROR(b->AddTensor(t, &node));
      sequence_lengths.emplace_back(node);
    }

    AttrValue parallel_copy;
    b->BuildAttrValue(parallel_copy_, &parallel_copy);

    AttrValue output_types;
    b->BuildAttrValue(output_dtypes(), &output_types);

    AttrValue N;
    b->BuildAttrValue<int64_t>(sequence_lengths_.size(), &N);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {{0, input_graph_node}, {1, batch_size}},  // Inputs
                      {{2, sequence_lengths}},  // List inputs.
                      {{kParallelCopy, parallel_copy},
                       {kToutputTypes, output_types},
                       {kNumPaddedShapes, N}},  // Attrs.
                      output));
    return OkStatus();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}

    Status Initialize(IteratorContext* ctx) override {
      {
        mutex_lock l(mu_);
        left_over_element_.reset();
      }
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      // PackedBatch collects elements (from the input dataset) that should go
      // into the batch. Once the PackedBatch is full we construct the actual
      // output tensors.
      PackedBatch packed_batch(dataset()->batch_size_,
                               dataset()->sequence_lengths_);

      // Iterate over the input dataset and add elements to packed_batch until
      // the next element doesn't fit.
      {
        // We only know that a batch is full if the next element doesn't fit. We
        // store the element as left over element and add it to the next batch.
        // This should always.
        if (left_over_element_.has_value()) {
          if (!packed_batch.TryAddElement(*left_over_element_)) {
            return errors::InvalidArgument(
                "Element didn't fit in empty batch. Please file a bug.");
          }
          left_over_element_.reset();
        }
        mutex_lock l(mu_);
        if (!input_impl_) {
          *end_of_sequence = true;
          return OkStatus();
        }
        *end_of_sequence = false;
        while (!*end_of_sequence) {
          Element element;
          TF_RETURN_IF_ERROR(
              input_impl_->GetNext(ctx, &element, end_of_sequence));
          if (!*end_of_sequence) {
            // Verify input element has valid shape. The sequence dimension is
            // usually dynamic, so we cannot perform the check statically.
            if (element.size() != dataset()->sequence_lengths_.size()) {
              return errors::InvalidArgument(
                  "Element had ", element.size(), " components but expected ",
                  dataset()->sequence_lengths_.size(), " components.");
            }
            for (int i = 0; i < element.size(); ++i) {
              if (element[i].dims() > 0 &&
                  element[i].dim_size(0) > dataset()->sequence_lengths_[i]) {
                return errors::InvalidArgument(
                    "Component ", i, " had sequence length ",
                    element[i].dim_size(0),
                    " which is larger than the sequence lengths for packing (",
                    dataset()->sequence_lengths_[i],
                    "). Please crop longer sequence before packing.");
              }
            }

            // Try adding the dataset to the batch. Break out of the loop if
            // the batch is full and store the left over element for the next
            // batch.
            if (!packed_batch.TryAddElement(element)) {
              left_over_element_ = std::make_optional(std::move(element));
              break;
            }
          }
        }
        if (*end_of_sequence) {
          input_impl_.reset();
        }
      }

      if (packed_batch.Empty()) {
        DCHECK(*end_of_sequence);
        return OkStatus();
      }

      TF_RETURN_IF_ERROR(CopyBatch(ctx, packed_batch, out_tensors));
      *end_of_sequence = false;
      return OkStatus();
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeUnknownRatioNode(std::move(args));
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock l(mu_);
      if (input_impl_)
        TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      else
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kExhausted), ""));
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock l(mu_);
      if (reader->Contains(full_name(kExhausted))) {
        input_impl_.reset();
      } else {
        TF_RETURN_IF_ERROR(
            dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
        TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      }
      return OkStatus();
    }

   private:
    // Create a tensor of zeros of the output component.
    Status ZerosLike(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                     const int64_t output_component) {
      const auto dtype = output_dtypes()[output_component];
      TensorShape shape;
      if (!output_shapes()[output_component].AsTensorShape(&shape)) {
        return errors::InvalidArgument(
            "Failed to get tensor shape for output component ",
            output_component);
      }
      out_tensors->emplace_back(ctx->allocator({}), dtype, shape);
      const CPUDevice* d = ctx->flr()->device()->eigen_cpu_device();
      Tensor& t = out_tensors->back();

#define HANDLE_TYPE(T)                                        \
  case DataTypeToEnum<T>::value:                              \
    functor::SetZeroFunctor<CPUDevice, T>()(*d, t.flat<T>()); \
    break;

      switch (dtype) {
        TF_CALL_POD_TYPES(HANDLE_TYPE);
        TF_CALL_tstring(HANDLE_TYPE);
#undef HANDLE_TYPE
        default:
          return errors::Unimplemented("Dtype ", dtype,
                                       " for output component ",
                                       output_component, " not supported.");
      }
      return OkStatus();
    }

    // Copy the elements from packed_batch into out_tensors (one entry per
    // output component). For each input component we have 3 output components:
    // - Packed and padded values.
    // - Segment IDs.
    // - Positions.
    Status CopyBatch(IteratorContext* ctx, PackedBatch& packed_batch,
                     std::vector<Tensor>* out_tensors) {
      out_tensors->reserve(output_shapes().size());
      if (packed_batch.Empty()) {
        return errors::InvalidArgument(
            "Tried to copy empty batch. Please file a bug.");
      }
      for (size_t input_component = 0;
           input_component < packed_batch.NumComponents(); ++input_component) {
        const auto output_component = 3 * input_component;
        TF_RETURN_IF_ERROR(ZerosLike(ctx, out_tensors, output_component));
        TF_RETURN_IF_ERROR(ZerosLike(ctx, out_tensors, output_component + 1));
        TF_RETURN_IF_ERROR(ZerosLike(ctx, out_tensors, output_component + 2));
        Tensor& values_t = (*out_tensors)[output_component];
        Tensor& segment_ids_t = (*out_tensors)[output_component + 1];
        Tensor& positions_t = (*out_tensors)[output_component + 2];

        // Build the output tuple component by copying one slice from each
        // input
        // element in the batch.
        auto copy_element_fn = [input_component, &packed_batch, &values_t,
                                &segment_ids_t, &positions_t](int index) {
          TF_RETURN_IF_ERROR(
              packed_batch.InsertValues(&values_t, index, input_component));
          TF_RETURN_IF_ERROR(packed_batch.AddSegmentIds(&segment_ids_t, index,
                                                        input_component));
          TF_RETURN_IF_ERROR(
              packed_batch.AddPositions(&positions_t, index, input_component));
          return OkStatus();
        };

        if (dataset()->parallel_copy_) {
          BlockingCounter counter(packed_batch.NumElements());
          Status status;
          mutex status_mu;
          const auto num_threads = ctx->runner_threadpool_size();
          const auto slice_size = packed_batch.NumElements() / num_threads;
          int64_t offset = 0;
          for (size_t i = 0; i < num_threads; ++i) {
            int64_t length = slice_size;
            // When the number of threads does not divide the number of elements
            // evenly, the size of some slices is incremented to guarantee their
            // sizes add up to the total number of elements.
            if (i < packed_batch.NumElements() % num_threads) {
              ++length;
            }
            (*ctx->runner())([offset, length, &status, &status_mu, &counter,
                              &copy_element_fn]() {
              for (size_t j = offset; j < offset + length; ++j) {
                {
                  Status s = copy_element_fn(j);
                  mutex_lock l(status_mu);
                  status.Update(s);
                }
                counter.DecrementCount();
              }
            });
            offset += length;
          }
          counter.Wait();
          TF_RETURN_IF_ERROR(status);
        } else {
          for (size_t i = 0; i < packed_batch.NumElements(); ++i) {
            TF_RETURN_IF_ERROR(copy_element_fn(i));
          }
        }
      }
      return OkStatus();
    }

    mutex mu_;
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(mu_);
    std::optional<Element> left_over_element_;
  };  // BatchAndPackDatasetOp::Dataset::Iterator

  const int64_t batch_size_;
  const bool parallel_copy_;
  const std::vector<int64_t> sequence_lengths_;
  const DatasetBase* const input_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};  // BatchAndPackDatasetOp::Dataset

void BatchAndPackDatasetOp::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase* input,
                                        DatasetBase** output) {
  int64_t batch_size;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64_t>(ctx, kBatchSize, &batch_size));
  OP_REQUIRES(ctx, batch_size > 0,
              errors::InvalidArgument("Batch size must be greater than zero."));

  OpInputList sequence_length_tensors;
  OP_REQUIRES_OK(ctx,
                 ctx->input_list(kSequenceLengths, &sequence_length_tensors));
  std::vector<int64_t> sequence_lengths;
  sequence_lengths.reserve(sequence_length_tensors.size());
  OP_REQUIRES(ctx,
              sequence_length_tensors.size() == input->output_shapes().size(),
              errors::InvalidArgument("Number of sequence lengths (",
                                      sequence_length_tensors.size(),
                                      ") must match the number of components "
                                      "in the input dataset's elements (",
                                      input->output_shapes().size(), ")"));
  for (const Tensor& sequence_length_t : sequence_length_tensors) {
    sequence_lengths.push_back(sequence_length_t.scalar<int64_t>()());
  }

  *output = new Dataset(ctx, batch_size, parallel_copy_,
                        std::move(sequence_lengths), input);
}

REGISTER_KERNEL_BUILDER(Name("BatchAndPackDataset").Device(DEVICE_CPU),
                        BatchAndPackDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
