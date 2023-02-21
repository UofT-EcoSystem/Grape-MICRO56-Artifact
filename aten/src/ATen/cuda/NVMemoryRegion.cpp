#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#if defined(BUILD_OUT_OF_PYTORCH_TREE)
#include <quik_fix/logging.h>
#else
#include <dmlc/logging.h>
#define QF_LOG_INFO LOG(INFO)
#define QF_CHECK CHECK
#endif

#include "NVMemoryRegion.h"

namespace at {
namespace cuda {

// constexpr int _C_NTHREADS_PER_BLOCK = 128;

// __launch_bounds__(_C_NTHREADS_PER_BLOCK) static __global__
//     void _cudaGetTrailingZerosStartingPos(
//         const uint32_t* const __restrict__ data,
//         const size_t nelems,
//         unsigned long long* const __restrict__ trailing_zeros_start_pos) {
//   const unsigned int g_threadIdx = threadIdx.x + blockDim.x * blockIdx.x;

//   if (g_threadIdx < nelems &&
//       // In the case when
//       //
//       //     nelems - 1 - gthreadIdx <= sTrailingZerosStartPos
//       //
//       // there is no point in doing the updates.
//       (nelems - 1 - g_threadIdx) > *trailing_zeros_start_pos) {
//     uint32_t local_data = data[nelems - 1 - g_threadIdx];
//     if (__any_sync(0xffffffff, local_data != 0)) {
//       if (threadIdx.x % warpSize == 0) {
//         atomicMax(
//             trailing_zeros_start_pos,
//             static_cast<unsigned long long>(nelems - 1 - g_threadIdx + 1));
//       }
//     }
//   }
// }

size_t getZeroCompressedUInt32SizeForHostMemoryRegion(
    HostMemoryRegion_t&& host_memory_region,
    const size_t size) {
  const uint32_t* const uint32_arr_data =
      static_cast<const uint32_t*>(host_memory_region.get());
  size_t uint32_arr_idx = size / sizeof(uint32_t);
  for (; uint32_arr_idx != 0; --uint32_arr_idx) {
    if (uint32_arr_data[uint32_arr_idx - 1]) {
      break;
    }
  }
  return uint32_arr_idx;
}

size_t NVMemoryRegion::getZeroCompressedUInt32Size() const {
  // unsigned long long trailing_zeros_start_pos_host,
  //     *trailing_zeros_start_pos_dev;

  // checkCudaErrors(
  //     cudaMalloc(&trailing_zeros_start_pos_dev, sizeof(unsigned long long)));
  // checkCudaErrors(
  //     cudaMemset(trailing_zeros_start_pos_dev, 0, sizeof(unsigned long
  //     long)));
  // _cudaGetTrailingZerosStartingPos<<<
  //     (size / sizeof(uint32_t) + _C_NTHREADS_PER_BLOCK - 1) /
  //         _C_NTHREADS_PER_BLOCK,
  //     _C_NTHREADS_PER_BLOCK>>>(
  //     static_cast<const uint32_t*>(dptr),
  //     size / sizeof(uint32_t),
  //     trailing_zeros_start_pos_dev);
  // checkCudaErrors(cudaDeviceSynchronize());
  // checkCudaErrors(cudaMemcpy(
  //     &trailing_zeros_start_pos_host,
  //     trailing_zeros_start_pos_dev,
  //     sizeof(unsigned long long),
  //     cudaMemcpyDeviceToHost));
  // checkCudaErrors(cudaFree(trailing_zeros_start_pos_dev));
  // return trailing_zeros_start_pos_host;
  QF_CHECK(size % sizeof(uint32_t) == 0);
  HostMemoryRegion_t host_memory_region = toCPU();
  return getZeroCompressedUInt32SizeForHostMemoryRegion(
      std::move(host_memory_region), size);
}

NVMemoryRegion::NVMemoryRegion(const size_t size_in_bytes) {
  checkCudaErrors(cudaMalloc(&dptr, size_in_bytes));
  size = size_in_bytes;
  _free_needed = true;
}

NVMemoryRegion::~NVMemoryRegion() {
  if (_free_needed) {
    cudaFree(dptr);
    dptr = nullptr;
  }
}

NVMemoryRegion::NVMemoryRegion(NVMemoryRegion&& other) noexcept {
  dptr = other.dptr;
  size = other.size;
  _free_needed = other._free_needed;
  other.dptr = nullptr;
  other.size = 0;
  other._free_needed = false;
}

NVMemoryRegion& NVMemoryRegion::operator=(NVMemoryRegion&& other) noexcept {
  dptr = other.dptr;
  size = other.size;
  _free_needed = other._free_needed;
  other.dptr = nullptr;
  other.size = 0;
  other._free_needed = false;
  return *this;
}

HostMemoryRegion_t NVMemoryRegion::toCPU(const bool fetch_from_gpu) const {
  if (!size) {
    return HostMemoryRegion_t(nullptr, &free);
  }
  HostMemoryRegion_t host_ptr(malloc(size), &free);
  if (fetch_from_gpu) {
    checkCudaErrors(
        cudaMemcpy(host_ptr.get(), dptr, size, cudaMemcpyDeviceToHost));
  }
  return host_ptr;
}

void NVMemoryRegion::dump(
    const std::string& filename,
    const bool as_plain_text,
    const size_t width) const {
  HostMemoryRegion_t host_ptr = toCPU();
  std::ofstream fout(
      filename + (as_plain_text ? ".txt" : ".bin"),
      as_plain_text ? std::ios::out : (std::ios::out | std::ios::binary));

  QF_LOG_INFO << "Dumping " << size << " bytes from the memory region";
  if (as_plain_text) {
    uint32_t* host_uint32_ptr = static_cast<uint32_t*>(host_ptr.get());
    QF_CHECK(size % sizeof(uint32_t) == 0)
        << "size=" << size << " %% sizeof(uint32_t) != 0";
    for (size_t i = 0; i < size / sizeof(uint32_t);) {
      for (size_t j = 0; i < size / sizeof(uint32_t) && j < width; ++j, ++i) {
        fout << std::hex << host_uint32_ptr[i] << std::dec;
      }
      fout << std::endl;
    }
    fout << std::endl;
  } else {
    fout.write(static_cast<const char*>(host_ptr.get()), size);
  }
  fout.close();
}

void NVMemoryRegion::load(const std::string& filename) {
  if (!size) {
    return;
  }
  HostMemoryRegion_t host_ptr = toCPU();
  std::ifstream fin(filename + ".bin", std::ios::in | std::ios::binary);

  QF_LOG_INFO << "Loading " << size << " bytes into the memory region";
  if (!fin) {
    QF_LOG_INFO << "Failed to load into the memory region";
    fin.close();
    return;
  }
  fin.read(static_cast<char*>(host_ptr.get()), size);
  checkCudaErrors(
      cudaMemcpy(dptr, host_ptr.get(), size, cudaMemcpyHostToDevice));
  fin.close();
}

} // namespace cuda
} // namespace at
