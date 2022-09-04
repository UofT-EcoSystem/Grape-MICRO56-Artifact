// <bojian/DynamicCUDAGraph>

#include "CUDAGlobalExecMask.cuh"
#include "CUDAGlobalExecMask.h"

#include <ATen/core/Formatting.h>
#include <c10/cuda/CUDAStream.h>

#include <dmlc/logging.h>
#include <dmlc/parameter.h>

#include <helper_cuda.h>

namespace at {
namespace cuda {

CUDAGlobalExecMask gCUDAGraphGlobalExecMask(true);

std::ostream& operator<<(
    std::ostream& out,
    const CUDAGlobalExecMask& exec_mask) {
  out << "ExecMask{mask=" << exec_mask.mask
      << ", outer_scope_mask=" << exec_mask.outer_scope_mask
      << ", input_ind=" << exec_mask.input_ind << "}";
  return out;
}

CUDAGlobalExecMask::CUDAGlobalExecMask(const bool initialize) {
  if (initialize) {
    checkCudaErrors(cudaMalloc(&mask, sizeof(bool)));
    checkCudaErrors(cudaMalloc(&outer_scope_mask, sizeof(bool)));
    checkCudaErrors(cudaMalloc(&input_ind, sizeof(bool)));
    checkCudaErrors(cudaMemset(mask, true, sizeof(bool)));
    checkCudaErrors(cudaMemset(outer_scope_mask, true, sizeof(bool)));
    checkCudaErrors(cudaMemset(input_ind, true, sizeof(bool)));
  }
}

void EnterGlobalExecMask(
    CUDAGlobalExecMask& outer_scope_exec_mask,
    at::Tensor curr_scope_mask,
    at::Tensor input_ind) {
  // make a copy of the current execution mask
  outer_scope_exec_mask.copy(gCUDAGraphGlobalExecMask);
  gCUDAGraphGlobalExecMask.outer_scope_mask = gCUDAGraphGlobalExecMask.mask;
  gCUDAGraphGlobalExecMask.mask =
      static_cast<bool*>(curr_scope_mask.data_ptr());
  gCUDAGraphGlobalExecMask.input_ind = static_cast<bool*>(input_ind.data_ptr());
}

void ExitGlobalExecMask(CUDAGlobalExecMask& outer_scope_exec_mask) {
  gCUDAGraphGlobalExecMask.copy(outer_scope_exec_mask);
}

} // namespace cuda
} // namespace at
