// <bojian/Grape>
#pragma once

#define GRAPE_CUDA_GLOBAL_INDICATOR 1

#if GRAPE_CUDA_GLOBAL_INDICATOR
#define GRAPE_GLOBAL_INDICATOR_KERNEL_ARGS                   \
  , bool* const __restrict__ current_scope_global_ind_value, \
      bool* const __restrict__ outer_scope_global_ind_value, \
      const bool* const __restrict__ input_ind_value
#else
#define GRAPE_GLOBAL_INDICATOR_KERNEL_ARGS
#endif

#if GRAPE_CUDA_GLOBAL_INDICATOR
#define GRAPE_UPDATE_GLOBAL_INDICATOR                         \
  if (blockIdx.x == 0 && threadIdx.x == 0) {                  \
    *current_scope_global_ind_value =                         \
        (*outer_scope_global_ind_value) & (*input_ind_value); \
  }                                                           \
  if ((*outer_scope_global_ind_value) & (*input_ind_value))
#else
#define GRAPE_UPDATE_GLOBAL_INDICATOR
#endif

#if GRAPE_CUDA_GLOBAL_INDICATOR
#define GRAPE_GLOBAL_INDICATOR_KERNEL_LAUNCH_ARGS                    \
  , ::at::cuda::gCUDAGlobalIndicator.current_scope_global_ind_value, \
      ::at::cuda::gCUDAGlobalIndicator.outer_scope_global_ind_value, \
      ::at::cuda::gCUDAGlobalIndicator.input_ind_value
#else
#define GRAPE_GLOBAL_INDICATOR_KERNEL_LAUNCH_ARGS
#endif

namespace at {
namespace cuda {

struct CUDAGlobalIndicator {
  bool* current_scope_global_ind_value;
  bool* outer_scope_global_ind_value;
  bool* input_ind_value;

  /// @brief Create a global indicator.
  /// @param initialize Whether to initialize the indicator. Only the primal
  /// indicator is initialized.
  explicit CUDAGlobalIndicator(const bool initialize = false);

  /// @brief Copy deeply from another indicator.
  /// @param other
  void copy(const CUDAGlobalIndicator& other);
};

extern CUDAGlobalIndicator gCUDAGlobalIndicator;

} // namespace cuda
} // namespace at
