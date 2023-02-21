#pragma once

#include <ATen/Tensor.h>

namespace at {
namespace cuda {

// defined in ./CUDAGlobalIndicator.cuh
struct CUDAGlobalIndicator;

/// @brief Enter function of a new global indicator scope.
/// @param outer_scope_global_ind The global indicator from the outer scope
/// @param current_scope_global_ind_value The indicator of the current scope.
/// Its value is to be assigned during the next kernel invocation to become
/// @c outer_scope_global_ind & @c input_ind_value .
/// @param input_ind_value The input indicator
void EnterCUDAGlobalIndicatorScope(
    CUDAGlobalIndicator& outer_scope_global_ind,
    Tensor current_scope_global_ind_value,
    Tensor input_ind_value);

/// @brief Exit function of the current indicator scope.
/// @param outer_scope_global_ind
void ExitCUDAGlobalIndicatorScope(
    const CUDAGlobalIndicator& outer_scope_global_ind);

/// @brief Temporarily stash the current CUDA global indicator.
/// @param outer_scope_global_ind
void EnterConstTrueCUDAGlobalIndicatorScope(
    CUDAGlobalIndicator& outer_scope_global_ind);

/// @brief Restore the previously stashed CUDA global indicator.
/// @param outer_scope_global_ind
void ExitConstTrueCUDAGlobalIndicatorScope(
    const CUDAGlobalIndicator& outer_scope_global_ind);

/// @brief Copy the data pointer to the scoreboard.
/// @param scoreboard_items
/// @param index
/// @param data_ptr
void BeamHypotheses_copyDataPtr(
    Tensor scoreboard_items,
    const size_t index,
    const size_t data_ptr);

/// @brief Force copying the value from the source to the destination,
/// by-passing the indicator.
/// @param dst
/// @param src
void forceMemcpy(
    const size_t dst,
    const size_t src,
    const size_t size_in_bytes);

/// @brief Force setting the value of the destination, by-passing the indicator.
/// @param dst
/// @param value
/// @param size_in_bytes
void forceMemset(const size_t dst, const int value, const size_t size_in_bytes);

} // namespace cuda
} // namespace at
