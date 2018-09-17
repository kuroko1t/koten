#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#define EIGEN_USE_THREADS
#include "tensorflow/stream_executor/stream.h"

#include <mpi.h>

class TFTensor {
public:
  //TFTensor(&Tensor);
 TFTensor(tensorflow::Tensor& tensor) : tensor_(tensor) {}
  virtual const MPI_Datatype dtype() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual ~TFTensor() = default;
 protected:
  tensorflow::Tensor tensor_;
};

struct kotenMPI {
  int rank = 0;
};

bool EnqueueTensorAllreduce(std::shared_ptr<tensorflow::Tensor> tensor,
                            std::shared_ptr<tensorflow::Tensor> output);

bool EnqueueTensorBroadcast(std::shared_ptr<tensorflow::Tensor> tensor,
                            std::shared_ptr<tensorflow::Tensor> output, int root_rank);
int horovod_rank();
