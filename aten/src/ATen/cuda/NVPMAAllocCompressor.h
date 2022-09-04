// <bojian/DynamicCUDAGraph>
#pragma once

#include <cstdlib>

#include <helper_cuda.h>

namespace at {
namespace cuda {

struct ZeroCompressedPtr {
  void* dptr = nullptr;
  size_t compressed_size = 0;

  ZeroCompressedPtr() = default;
  ZeroCompressedPtr(void* const dptr, const size_t compressed_size)
      : dptr(dptr), compressed_size(compressed_size) {}
  ZeroCompressedPtr(const ZeroCompressedPtr&) = delete;
  ZeroCompressedPtr(ZeroCompressedPtr&&);
  ZeroCompressedPtr& operator=(const ZeroCompressedPtr&) = delete;
  ZeroCompressedPtr& operator=(ZeroCompressedPtr&& other);
  ~ZeroCompressedPtr();

  bool NotNull() const {
    return dptr != nullptr && compressed_size != 0;
  }
};

struct DecompressEngine {
  cudaStream_t stream;

  DecompressEngine() {
    checkCudaErrors(cudaStreamCreate(&stream));
  }
  ~DecompressEngine() {
    try {
      checkCudaErrors(cudaStreamDestroy(stream));
    } catch (...) {
    }
  }
};

} // namespace cuda
} // namespace at
