#include <iostream>
#include <memory>
#include "../tensorflow/mpi_ops.h"

bool EnqueueTensorBroadcast(std::shared_ptr<tensorflow::Tensor> tensor,
                            std::shared_ptr<tensorflow::Tensor> output, int root_rank) {
  return true;
}

bool EnqueueTensorAllreduce(std::shared_ptr<tensorflow::Tensor> tensor,
                            std::shared_ptr<tensorflow::Tensor> output) {
  return true;
}
