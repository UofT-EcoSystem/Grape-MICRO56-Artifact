#include <helper_cuda.h>

#include "CUDAGlobalIndicator.cuh"
#include "CUDAGlobalIndicator.h"
#include "CUDAGraph.h"

#include <dmlc/logging.h>

namespace at {
namespace cuda {

CUDAGlobalIndicator gCUDAGlobalIndicator(true);

std::ostream& operator<<(std::ostream& out, const CUDAGlobalIndicator& ind) {
  bool current_scope_global_ind_value, outer_scope_global_ind_value,
      input_ind_value;
  checkCudaErrors(cudaMemcpy(
      &current_scope_global_ind_value,
      ind.current_scope_global_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(
      &outer_scope_global_ind_value,
      ind.outer_scope_global_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(
      &input_ind_value,
      ind.input_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToHost));
  out << "GlobalIndicator{current_scope_global_ind @"
      << ind.current_scope_global_ind_value << " (" << std::boolalpha
      << current_scope_global_ind_value << "), outer_scope_global_ind @"
      << ind.outer_scope_global_ind_value << " ("
      << outer_scope_global_ind_value << "), input_ind @" << ind.input_ind_value
      << " (" << input_ind_value << std::noboolalpha << ")}";
  return out;
}

CUDAGlobalIndicator::CUDAGlobalIndicator(const bool initialize) {
  if (initialize) {
    checkCudaErrors(cudaMalloc(&current_scope_global_ind_value, sizeof(bool)));
    checkCudaErrors(cudaMalloc(&outer_scope_global_ind_value, sizeof(bool)));
    checkCudaErrors(cudaMalloc(&input_ind_value, sizeof(bool)));
    checkCudaErrors(
        cudaMemset(current_scope_global_ind_value, true, sizeof(bool)));
    checkCudaErrors(
        cudaMemset(outer_scope_global_ind_value, true, sizeof(bool)));
    checkCudaErrors(cudaMemset(input_ind_value, true, sizeof(bool)));
  }
}

void CUDAGlobalIndicator::copy(const CUDAGlobalIndicator& other) {
  checkCudaErrors(cudaMemcpy(
      current_scope_global_ind_value,
      other.current_scope_global_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(
      outer_scope_global_ind_value,
      other.outer_scope_global_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(
      input_ind_value,
      other.input_ind_value,
      sizeof(bool),
      cudaMemcpyDeviceToDevice));
}

void EnterCUDAGlobalIndicatorScope(
    CUDAGlobalIndicator& outer_scope_global_ind,
    at::Tensor current_scope_global_ind_value,
    at::Tensor input_ind_value) {
  // Make a copy of the current global indicator. Note that we deliberately
  // invoke the shadow copy operator here.
  outer_scope_global_ind = gCUDAGlobalIndicator;
  gCUDAGlobalIndicator.outer_scope_global_ind_value =
      gCUDAGlobalIndicator.current_scope_global_ind_value;
  gCUDAGlobalIndicator.current_scope_global_ind_value =
      static_cast<bool*>(current_scope_global_ind_value.data_ptr());
  gCUDAGlobalIndicator.input_ind_value =
      static_cast<bool*>(input_ind_value.data_ptr());
}

void ExitCUDAGlobalIndicatorScope(
    const CUDAGlobalIndicator& outer_scope_global_ind) {
  gCUDAGlobalIndicator = outer_scope_global_ind;
}

void EnterConstTrueCUDAGlobalIndicatorScope(
    CUDAGlobalIndicator& outer_scope_global_ind) {
  LOG(INFO) << "Temporarily stashing the current " << gCUDAGlobalIndicator;
  outer_scope_global_ind.copy(gCUDAGlobalIndicator);
  LOG(INFO) << "  on " << outer_scope_global_ind;
  checkCudaErrors(cudaMemset(
      gCUDAGlobalIndicator.current_scope_global_ind_value, true, sizeof(bool)));
  checkCudaErrors(cudaMemset(
      gCUDAGlobalIndicator.outer_scope_global_ind_value, true, sizeof(bool)));
  checkCudaErrors(
      cudaMemset(gCUDAGlobalIndicator.input_ind_value, true, sizeof(bool)));
}

void ExitConstTrueCUDAGlobalIndicatorScope(
    const CUDAGlobalIndicator& outer_scope_global_ind) {
  LOG(INFO) << "Restoring the previously stashed " << outer_scope_global_ind;
  gCUDAGlobalIndicator.copy(outer_scope_global_ind);
}

__global__ void cudaGraphConditionallyLaunchOnDevice(
    const cudaGraphExec_t graph_exec,
    bool* sync_barrier GRAPE_GLOBAL_INDICATOR_KERNEL_ARGS) {
  GRAPE_UPDATE_GLOBAL_INDICATOR {
    volatile bool* volatile_sync_barrier = sync_barrier;
    *volatile_sync_barrier = true;
    cudaGraphLaunch(graph_exec, cudaStreamGraphFireAndForget);
    while (*volatile_sync_barrier) {
    }
  }
}

void instantiateCUDAGraphOnDeviceV2(CUDAGraph& graph) {
  graph.capture_end_epilog(/*instantiate_on_device=*/true);
}

void embedDeviceCUDAGraph(CUDAGraph& graph, Tensor sync_barrier) {
  // LOG(INFO) << "Embedding the device CUDAGraph";
  CHECK(graph.has_graph_exec_) << "The CUDAGraph must be instantiated before";

  cudaGraphConditionallyLaunchOnDevice<<<
      1,
      1,
      0,
      getCurrentCUDAStream().stream()>>>(
      graph.graph_exec_,
      static_cast<bool*>(sync_barrier.data_ptr())
          GRAPE_GLOBAL_INDICATOR_KERNEL_LAUNCH_ARGS);
}

__global__ void BeamHypotheses_cudaDataPtrCopy(
    uint64_t* const scoreboard_items,
    const size_t index,
    const size_t data_ptr GRAPE_GLOBAL_INDICATOR_KERNEL_ARGS) {
  GRAPE_UPDATE_GLOBAL_INDICATOR {
    scoreboard_items[index] = data_ptr;
  }
}

void BeamHypotheses_copyDataPtr(
    Tensor scoreboard_items,
    const size_t index,
    const size_t data_ptr) {
  BeamHypotheses_cudaDataPtrCopy<<<1, 1, 0, getCurrentCUDAStream().stream()>>>(
      static_cast<uint64_t*>(scoreboard_items.data_ptr()),
      index,
      data_ptr GRAPE_GLOBAL_INDICATOR_KERNEL_LAUNCH_ARGS);
}

void forceMemcpy(
    const size_t dst,
    const size_t src,
    const size_t size_in_bytes) {
  checkCudaErrors(cudaMemcpyAsync(
      reinterpret_cast<void*>(dst),
      reinterpret_cast<void*>(src),
      size_in_bytes,
      cudaMemcpyDeviceToDevice,
      getCurrentCUDAStream().stream()));
}

void forceMemset(
    const size_t dst,
    const int value,
    const size_t size_in_bytes) {
  checkCudaErrors(cudaMemsetAsync(
      reinterpret_cast<void*>(dst),
      value,
      size_in_bytes,
      getCurrentCUDAStream().stream()));
}

} // namespace cuda
} // namespace at
