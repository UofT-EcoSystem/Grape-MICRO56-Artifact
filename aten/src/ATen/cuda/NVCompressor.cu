#include <iomanip>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#if !defined(BUILD_OUT_OF_PYTORCH_TREE)
#include <c10/cuda/CUDAStream.h>
#endif
#include "NVCompressor.h"

#if defined(BUILD_OUT_OF_PYTORCH_TREE)
#include <quik_fix/logging.h>
#else
#include <dmlc/logging.h>
#define QF_LOG_INFO LOG(INFO)
#define QF_CHECK CHECK
#endif

namespace at {
namespace cuda {

ZeroCompressedRegion ZeroCompressedRegion::compress(
    const NVMemoryRegion& orig_region) {
  if (!orig_region.size) {
    return ZeroCompressedRegion();
  }

  const size_t zero_compressed_uint32_size =
                   orig_region.getZeroCompressedUInt32Size(),
               zero_compressed_size =
                   zero_compressed_uint32_size * sizeof(uint32_t);

  QF_LOG_INFO << "Compressed size=" << zero_compressed_size
              << " (=" << (zero_compressed_size * 1.0 / (1024 * 1024)) << " MB)"
              << " => " << std::setprecision(2)
              << orig_region.size * 1.0 / zero_compressed_size << "x reduction";
  ZeroCompressedRegion ret(zero_compressed_size);
  checkCudaErrors(cudaMemcpy(
      ret.dptr,
      orig_region.dptr,
      zero_compressed_size,
      cudaMemcpyDeviceToDevice));
  return ret;
}

void ZeroCompressedRegion::decompressTo(
    const NVMemoryRegion& orig_region,
    const cudaStream_t stream) const {
  if (size == 0) {
    return;
  }
  checkCudaErrors(cudaMemcpyAsync(
      orig_region.dptr, dptr, size, cudaMemcpyDeviceToDevice, stream));
  checkCudaErrors(cudaMemsetAsync(
      static_cast<char*>(orig_region.dptr) + size,
      0,
      orig_region.size - size,
      stream));
}

size_t RLECompressedRegion::page_size = 2048;

RLECompressedRegion RLECompressedRegion::compress(
    const NVMemoryRegion& orig_region) {
  if (!orig_region.size) {
    return RLECompressedRegion();
  }
  HostMemoryRegion_t host_memory_region = orig_region.toCPU();
  const size_t zero_compressed_uint32_orig_size =
      orig_region.getZeroCompressedUInt32Size();
  const size_t uint32s_per_page =
      RLECompressedRegion::page_size / sizeof(uint32_t);
  const size_t zero_compressed_uint32_size =
      (zero_compressed_uint32_orig_size + uint32s_per_page - 1) /
      uint32s_per_page * uint32s_per_page;

  const uint32_t* const uint32_arr_data =
      static_cast<const uint32_t*>(host_memory_region.get());
  const size_t num_pages = zero_compressed_uint32_size / uint32s_per_page;

  std::unique_ptr<uint32_t[]> compressed_data(
      new uint32_t[zero_compressed_uint32_size]);
  size_t rle_compressed_uint32_arr_size = 0;
  std::unique_ptr<uint32_t[]> compressed_data_offsets(new uint32_t[num_pages]);

  for (size_t page_idx = 0; page_idx < num_pages; ++page_idx) {
    compressed_data_offsets[page_idx] = rle_compressed_uint32_arr_size;
    for (size_t uint32_arr_idx_per_page = 0;
         uint32_arr_idx_per_page < uint32s_per_page;
         ++uint32_arr_idx_per_page) {
      uint32_t contig_count = 1;
      for (; uint32_arr_idx_per_page < (uint32s_per_page - 1) &&
           uint32_arr_data
                   [page_idx * uint32s_per_page + uint32_arr_idx_per_page] ==
               uint32_arr_data
                   [page_idx * uint32s_per_page + uint32_arr_idx_per_page + 1];
           ++uint32_arr_idx_per_page, ++contig_count) {
      }
      compressed_data[rle_compressed_uint32_arr_size] = uint32_arr_data
          [page_idx * uint32s_per_page + uint32_arr_idx_per_page];
      compressed_data[rle_compressed_uint32_arr_size + 1] = contig_count;
      rle_compressed_uint32_arr_size += 2;

    } // for (uint32_arr_idx_per_page in [0, uint32_arr_size_per_page))
  } // for (page_idx in [0, num_pages))

  size_t rle_compressed_size =
      rle_compressed_uint32_arr_size * sizeof(uint32_t);

  QF_LOG_INFO << "Compressed size=" << rle_compressed_size
              << " (=" << (rle_compressed_size * 1.0 / (1024 * 1024)) << " MB)"
              << " => " << std::setprecision(4)
              << zero_compressed_uint32_orig_size * 1.0 /
          (rle_compressed_uint32_arr_size)
              << "x reduction ("
              << zero_compressed_uint32_orig_size * 1.0 /
          (rle_compressed_uint32_arr_size + num_pages)
              << "x with offsets)";
  RLECompressedRegion ret(rle_compressed_size, num_pages);

  checkCudaErrors(cudaMemcpy(
      ret.dptr,
      compressed_data.get(),
      rle_compressed_size,
      cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(
      ret.metadata_page_offsets.dptr,
      compressed_data_offsets.get(),
      num_pages * sizeof(uint32_t),
      cudaMemcpyHostToDevice));
  return ret;
}

__global__ void cudaRLEDecompress(
    const uint32_t* __restrict__ const rle_compressed_uint32_arr,
    const uint32_t* __restrict__ const page_offsets,
    uint32_t* __restrict__ const orig_uint32_arr,
    const size_t num_pages,
    const size_t uint32s_per_page,
    const size_t rle_compressed_uint32_arr_size) {
  const unsigned g_threadIdx = threadIdx.x + blockDim.x * blockIdx.x;

  if (g_threadIdx < num_pages) {
    uint32_t rle_compressed_uint32_arr_idx = page_offsets[g_threadIdx],
             rle_compressed_uint32_arr_idx_end = g_threadIdx == num_pages - 1
        ? rle_compressed_uint32_arr_size
        : page_offsets[g_threadIdx + 1];

    size_t orig_uint32_arr_idx = g_threadIdx * uint32s_per_page;

    for (; rle_compressed_uint32_arr_idx < rle_compressed_uint32_arr_idx_end;
         rle_compressed_uint32_arr_idx += 2) {
      const uint32_t
          value = rle_compressed_uint32_arr[rle_compressed_uint32_arr_idx],
          contig_count =
              rle_compressed_uint32_arr[rle_compressed_uint32_arr_idx + 1];

      for (size_t contig_idx = 0; contig_idx < contig_count;
           ++contig_idx, ++orig_uint32_arr_idx) {
        orig_uint32_arr[orig_uint32_arr_idx] = value;
      }
    }
  } // if (g_threadIdx < num_pages)
}

void RLECompressedRegion::decompressTo(
    const NVMemoryRegion& orig_region,
    const cudaStream_t stream) const {
  if (size == 0) {
    return;
  }

  cudaRLEDecompress<<<(num_pages + 31) / 32, 32, 0, stream>>>(
      static_cast<const uint32_t*>(dptr),
      static_cast<const uint32_t*>(metadata_page_offsets.dptr),
      static_cast<uint32_t*>(orig_region.dptr),
      num_pages,
      RLECompressedRegion::page_size / sizeof(uint32_t),
      size / sizeof(uint32_t));
}

CompressEngine gCompressEngine;

CompressEngine::CompressEngine() {
  checkCudaErrors(cudaStreamCreate(&_working_stream));
  checkCudaErrors(
      cudaEventCreateWithFlags(&_workitem_in_progress, cudaEventDisableTiming));
}

CompressEngine::~CompressEngine() {
  cudaStreamDestroy(_working_stream);
  _working_stream = nullptr;
  cudaEventDestroy(_workitem_in_progress);
  _workitem_in_progress = nullptr;
}

void CompressEngine::waitForAllWorkitems() {
  checkCudaErrors(cudaStreamSynchronize(_working_stream));
}

void CompressEngine::waitForCurrentWorkitem() {
#if defined(BUILD_OUT_OF_PYTORCH_TREE)
  checkCudaErrors(cudaEventSynchronize(_workitem_in_progress));
#else
  checkCudaErrors(cudaStreamWaitEvent(
      ::c10::cuda::getCurrentCUDAStream().stream(), _workitem_in_progress));
#endif
}

} // namespace cuda
} // namespace at
