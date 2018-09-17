#include "mpi_ops.h"

//Status ConvertStatus(common::Status status) {
//  switch (status.type()) {
//  case common::OK:
//    return Status::OK();
//  case common::UNKNOWN_ERROR:
//    return errors::Unknown(status.reason());
//  case common::PRECONDITION_ERROR:
//    return errors::FailedPrecondition(status.reason());
//  case common::ABORTED:
//    return errors::Aborted(status.reason());
//  default:
//    return errors::Unknown("Unknown error.");
//  }
//}
//
//common::Status ConvertStatus(Status status) {
//  switch (status.code()) {
//  case error::Code::OK:
//    return common::Status::OK();
//  case error::Code::UNKNOWN:
//    return common::Status::UnknownError(status.error_message());
//  case error::Code::FAILED_PRECONDITION:
//    return common::Status::PreconditionError(status.error_message());
//  case error::Code::ABORTED:
//    return common::Status::Aborted(status.error_message());
//  default:
//    return common::Status::UnknownError("Unknown error.");
//  }
//}

//class Tensor {
//public:
//  Tensor(Tensor& tensor);
//  virtual const MPI_Datatype dtype() const override;
//  virtual const void* data() const override;
//  virtual int64_t size() const override;
//protected:
//  ::tensorflow::Tensor tensor_;
//};

//TFTensor::TFTensor(Tensor& tensor) : tensor_(tensor) {}

kotenMPI koten_mpi;

const MPI_Datatype TFTensor::dtype() const {
  switch (tensor_.dtype()) {
  case tensorflow::DT_UINT8:
    return MPI_UINT8_T;
  case tensorflow::DT_INT8:
    return MPI_INT8_T;
  case tensorflow::DT_UINT16:
    return MPI_UINT16_T;
  case tensorflow::DT_INT16:
    return MPI_INT16_T;
  case tensorflow::DT_INT32:
    return MPI_INT32_T;
  case tensorflow::DT_INT64:
    return MPI_INT64_T;
  case tensorflow::DT_FLOAT:
    return MPI_FLOAT;
  case tensorflow::DT_DOUBLE:
    return MPI_DOUBLE;
  case tensorflow::DT_BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

const void* TFTensor::data() const { return (const void*)tensor_.tensor_data().data(); }

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

int horovod_rank() {
  //if (!koten_mpi.initialization_done) {
  //  return -1;
  //}
  return koten_mpi.rank;
}


class KotenAllreduceOp : public tensorflow::AsyncOpKernel {
public:
  explicit KotenAllreduceOp(tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(tensorflow::OpKernelContext* context, DoneCallback done) override {
    //OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
    //                     done);

    auto node_name = name();
    auto tensor = context->input(0);
    tensorflow::Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(0, tensor.shape(), &output), done);
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    //auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<tensorflow::Tensor>(tensor);
    auto hvd_output = std::make_shared<tensorflow::Tensor>(*output);
    auto enqueue_result = EnqueueTensorAllreduce(
         hvd_tensor, hvd_output);
    //OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }
};

REGISTER_KERNEL_BUILDER(Name("KotenAllreduce").Device(tensorflow::DEVICE_CPU),
                        KotenAllreduceOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("KotenAllreduce").Device(DEVICE_GPU),
                        KotenAllreduceOp);
#endif

REGISTER_OP("KotenAllreduce")
    .Attr("T: {int32, int64, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");


class KotenBroadcastOp : public tensorflow::AsyncOpKernel {
public:
  explicit KotenBroadcastOp(tensorflow::OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
  }

  void ComputeAsync(tensorflow::OpKernelContext* context, DoneCallback done) override {
    //OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
    //                     done);

    auto node_name = name();
    auto tensor = context->input(0);
    tensorflow::Tensor* output = nullptr;
    if (horovod_rank() == root_rank_) {
      context->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
    }
    // ReadyEvent makes sure input tensor is ready, and output is allocated.
    //auto ready_event = std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
    //auto hvd_context = std::make_shared<TFOpContext>(context);
    auto hvd_tensor = std::make_shared<tensorflow::Tensor>(tensor);
    std::shared_ptr<tensorflow::Tensor> hvd_output = nullptr;
    if (output != nullptr) {
      hvd_output = std::make_shared<tensorflow::Tensor>(*output);
    }
    auto enqueue_result = EnqueueTensorBroadcast(
        hvd_tensor, hvd_output, root_rank_);
    //OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  }

private:
  int root_rank_;
};

REGISTER_KERNEL_BUILDER(Name("KotenBroadcast").Device(tensorflow::DEVICE_CPU),
                        KotenBroadcastOp);
#if HOROVOD_GPU_BROADCAST
REGISTER_KERNEL_BUILDER(Name("KotenBroadcast").Device(DEVICE_GPU),
                        KotenBroadcastOp);
#endif

REGISTER_OP("KotenBroadcast")
    .Attr(
        "T: {uint8, int8, uint16, int16, int32, int64, float32, float64, bool}")
    .Attr("root_rank: int")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Broadcast on a tensor. All other processes that do a broadcast
on a tensor with the same name must have the same dimension for that tensor.

Arguments
    tensor:     A tensor to broadcast.
    root_rank:  Rank that will send data, other ranks will receive data.

Output
    output:    A tensor with the same shape as `tensor` and same value as
               `tensor` on root rank.
)doc");
