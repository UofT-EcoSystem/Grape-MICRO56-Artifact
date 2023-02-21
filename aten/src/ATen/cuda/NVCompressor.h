#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "NVMemoryRegion.h"

namespace at {
namespace cuda {

struct ZeroCompressedRegion : NVMemoryRegion {
  ZeroCompressedRegion() = default;
  explicit ZeroCompressedRegion(const size_t compressed_size)
      : NVMemoryRegion(compressed_size) {}
  /// @name Compression/Decompression Subroutines
  /// @{
  static ZeroCompressedRegion compress(const NVMemoryRegion& region);
  void decompressTo(const NVMemoryRegion& region, const cudaStream_t stream)
      const;
  /// @}
};

struct RLECompressedRegion : NVMemoryRegion {
  size_t num_pages;
  NVMemoryRegion metadata_page_offsets;

  static size_t page_size;
  RLECompressedRegion() = default;
  RLECompressedRegion(const size_t compressed_size, const size_t num_pages)
      : NVMemoryRegion(compressed_size),
        num_pages(num_pages),
        metadata_page_offsets(num_pages * sizeof(uint32_t)) {}
  /// @name Compression/Decompression Subroutines
  /// @{
  static RLECompressedRegion compress(const NVMemoryRegion& region);
  void decompressTo(const NVMemoryRegion& region, const cudaStream_t stream)
      const;
  /// @}
};

class CompressEngine {
 private:
  // Keep a dedicated working stream for the engine, since most
  // compression/decompression works are memory operations that can in theory
  // happen in parallel with compute workloads.
  cudaStream_t _working_stream = nullptr;
  cudaEvent_t _workitem_in_progress = nullptr;

 public:
  CompressEngine();
  ~CompressEngine();
  CompressEngine(const CompressEngine&) = delete;
  CompressEngine(CompressEngine&&) = delete;
  CompressEngine& operator=(const CompressEngine&) = delete;
  CompressEngine& operator=(CompressEngine&&) = delete;

  /// @brief Generic interface for compressing a memory region
  /// @tparam TCompressedRegion
  /// @param orig_region The original memory region
  /// @return The compressed object
  template <typename TCompressedRegion>
  TCompressedRegion compress(const NVMemoryRegion& orig_region) {
    return TCompressedRegion::compress(orig_region);
  }

  /// @brief Generic interface for decompressing to a memory region
  /// @tparam TCompressedRegion
  /// @param compressed_region The compressed memory region
  /// @param orig_region The original memory region
  template <typename TCompressedRegion>
  void decompress(
      const TCompressedRegion& compressed_region,
      const NVMemoryRegion& orig_region) {
    compressed_region.decompressTo(orig_region, _working_stream);
    checkCudaErrors(cudaEventRecord(_workitem_in_progress, _working_stream));
  }

  /// @brief Have the current compute stream wait for the decompression to
  /// complete.
  void waitForAllWorkitems();

  /// @brief Have the current compute stream wait for the current decompression
  /// workitem to complete.
  void waitForCurrentWorkitem();
};

extern CompressEngine gCompressEngine;

} // namespace cuda
} // namespace at
