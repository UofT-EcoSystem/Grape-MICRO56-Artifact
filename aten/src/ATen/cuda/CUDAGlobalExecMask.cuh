// <bojian/DynamicCUDAGraph>

#pragma once

#define CUDA_GRAPH_GLOBAL_EXEC_MASK 1

#if CUDA_GRAPH_GLOBAL_EXEC_MASK
#define CUDA_GRAPH_GLOBAL_EXEC_MASK_KERNEL_ARGS       \
  , bool* const __restrict__ global_exec_mask,        \
      bool* const __restrict__ outer_scope_exec_mask, \
      const bool* const __restrict__ input_ind
#else
#define CUDA_GRAPH_GLOBAL_EXEC_MASK_KERNEL_ARGS
#endif

#if CUDA_GRAPH_GLOBAL_EXEC_MASK
#define UPDATE_GLOBAL_EXEC_MASK                                  \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                     \
    *global_exec_mask = (*outer_scope_exec_mask) & (*input_ind); \
  }                                                              \
  if ((*outer_scope_exec_mask) & (*input_ind))
#else
#define UPDATE_GLOBAL_EXEC_MASK
#endif

#if CUDA_GRAPH_GLOBAL_EXEC_MASK
#define CUDA_GRAPH_GLOBAL_EXEC_MASK_KERNEL_LAUNCH_ARGS       \
  , ::at::cuda::gCUDAGraphGlobalExecMask.mask,               \
      ::at::cuda::gCUDAGraphGlobalExecMask.outer_scope_mask, \
      ::at::cuda::gCUDAGraphGlobalExecMask.input_ind
#else
#define CUDA_GRAPH_GLOBAL_EXEC_MASK_KERNEL_LAUNCH_ARGS
#endif

namespace at {
namespace cuda {

struct CUDAGlobalExecMask {
  bool* mask;
  bool* outer_scope_mask;
  bool* input_ind;

  CUDAGlobalExecMask(const bool initialize = false);

  void copy(const CUDAGlobalExecMask& other) {
    mask = other.mask;
    outer_scope_mask = other.outer_scope_mask;
    input_ind = other.input_ind;
  }
};

extern CUDAGlobalExecMask gCUDAGraphGlobalExecMask;

} // namespace cuda
} // namespace at
