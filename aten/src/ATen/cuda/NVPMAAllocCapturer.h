#pragma once

#include <array>
#include <vector>

#include "../../../../../open-gpu-kernel-modules/src/nvidia/arch/nvalloc/unix/src/NVCapturePMAAllocMode.h"
#include "NVMemoryRegion.h"

#define NV_HUGE_PAGE_SIZE (2 * 1024 * 1024)

namespace at {
namespace cuda {

#if !defined(BUILD_OUT_OF_PYTORCH_TREE)
struct CUDAGraph;
#endif

// using NVPMAQueryResult_t =
//     std::tuple<NVCapturePMAAllocMode_t, size_t, std::vector<size_t>>;
struct NVPMAQueryResult_t {
  NVCapturePMAAllocMode_t current_mode;
  size_t current_residual_capacity;
  size_t current_residual_idx;
  std::vector<size_t> record_entries;
};

/// @brief The capturer captures a certain memory allocation and stores it
/// inside the shadow region.
class
#if !defined(BUILD_OUT_OF_PYTORCH_TREE)
    TORCH_CUDA_CPP_API
#endif
        NVPMAAllocCapturer {
 private:
  /// @name NVIDIA Driver Communication Subroutines
  /// @{
  static constexpr unsigned int _C_DEVICE_PCI_ID_LEN = 128;
  /// @brief Get the PCI bus ID of the current device, needed to derive the
  /// location of the @c capture_pma_alloc file.
  /// @return The PCI bus ID
  /// @sa @c _getNVDrvCapturePMAAllocAbsPathStr
  static std::array<char, _C_DEVICE_PCI_ID_LEN> _getCurrentDevicePCIBusId();
  /// @brief Get the absolute path to the @c capture_pma_alloc file in the
  /// NVIDIA driver.
  /// @return The absolute path
  static const std::string& _getNVDrvCapturePMAAllocAbsPathStr();
  /// @brief Change the mode of the capturer in the driver.
  /// @param capture_pma_alloc_mode
  /// @return True if the change is successful, false otherwise
  static bool _writeToDrvFile(
      const NVCapturePMAAllocMode_t capture_pma_alloc_mode);

 public:
  static void setToRecordMallocs();
  static void setToShadowMallocs();
  static void setToShadowNextMalloc();
  static void setToProbeNextMalloc();
  static void setToShadowNextMallocAndStashResiduals();
  static void setToShadowNextMallocAndAppendResiduals();
  static void setToShadowResiduals();
  static void clearListOfResiduals();
  static void setToRecordNextAndOverwrite();
  static void resetToDefault();

  static bool verbose;
  /// @brief Set the verbose level. This function is used at the Python level.
  /// @param verbose
  static void setVerbosity(const bool verbose);

  /// @brief Query the recorded PMA entries.
  /// @return A pair. The first is the number of residuals. The second is a list
  /// of memory allocation sizes.
  static NVPMAQueryResult_t queryRecordedPMAAllocSizes();
  /// @}

  /// @name Residuals
  /// @{
  static std::vector<NVMemoryRegion> residuals_snapshot;
  static void updateResidualsSnapshot();
  /// @}

  /// @name Utility Functions
  /// @{
  void dump(
      const std::string& filename,
      const bool as_plain_text = true,
      const size_t width = 128);
  void load(const std::string& filename);
  /// @}

  NVPMAAllocCapturer() = default;

 private:
  NVPMAAllocCapturer(const NVPMAAllocCapturer&) = delete;
  NVPMAAllocCapturer& operator=(const NVPMAAllocCapturer&) = delete;

 public:
  NVPMAAllocCapturer(NVPMAAllocCapturer&&) noexcept = default;
  NVPMAAllocCapturer& operator=(NVPMAAllocCapturer&&) noexcept = default;

  /// @name Memory Regions
  /// @{
  NVMemoryRegion shadow_main_region;

  /// @brief Materialize the memory allocations into the shadow regions.
  void materialize();
  /// @}

#if !defined(BUILD_OUT_OF_PYTORCH_TREE)
  friend struct CUDAGraph;
#endif
};

} // namespace cuda
} // namespace at

// Materialize the capturer using the memory allocations in the statement.
#define MATERIALIZE_CAPTURER_WITH_STMT(capturer, stmt)    \
  do {                                                    \
    ::at::cuda::NVPMAAllocCapturer::setToRecordMallocs(); \
    (stmt);                                               \
    (capturer).materialize();                             \
    ::at::cuda::NVPMAAllocCapturer::resetToDefault();     \
  } while (false)

// Probe the next immediate memory allocation. Since no memory allocations will
// take place, the statement is set to fail and `cudaGetLastError` is there to
// clear the failure status.
#define PROBE_NEXT_IMM_MALLOC_FROM_STMT(ret, stmt)                      \
  do {                                                                  \
    ::at::cuda::NVPMAAllocCapturer::setToProbeNextMalloc();             \
    stmt;                                                               \
    cudaGetLastError();                                                 \
    ret = ::at::cuda::NVPMAAllocCapturer::queryRecordedPMAAllocSizes(); \
    ::at::cuda::NVPMAAllocCapturer::resetToDefault();                   \
  } while (false)
