// <bojian/DynamicCUDAGraph>

#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace cuda {

struct CUDAGlobalExecMask;

void EnterGlobalExecMask(
    CUDAGlobalExecMask& outer_scope_exec_mask,
    Tensor current_scope_mask,
    Tensor input_ind);
void ExitGlobalExecMask(CUDAGlobalExecMask& outer_scope_exec_mask);

} // namespace cuda
} // namespace at
