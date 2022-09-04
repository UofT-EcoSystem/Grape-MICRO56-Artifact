// <bojian/DynamicCUDAGraph>
#include <helper_cuda.h>

#include "NVPMAAllocCompressor.h"

namespace at {
namespace cuda {

ZeroCompressedPtr::ZeroCompressedPtr(ZeroCompressedPtr&& other) {
  dptr = other.dptr;
  other.dptr = nullptr;
  compressed_size = other.compressed_size;
  other.compressed_size = 0;
  // residual_ptr = other.residual_ptr;
  // other.residual_ptr = nullptr;
  // compressed_residual_size = other.compressed_residual_size;
  // other.compressed_residual_size = 0;
}

ZeroCompressedPtr& ZeroCompressedPtr::operator=(ZeroCompressedPtr&& other) {
  dptr = other.dptr;
  other.dptr = nullptr;
  compressed_size = other.compressed_size;
  other.compressed_size = 0;
  // residual_ptr = other.residual_ptr;
  // other.residual_ptr = nullptr;
  // compressed_residual_size = other.compressed_residual_size;
  // other.compressed_residual_size = 0;
  return *this;
}

ZeroCompressedPtr::~ZeroCompressedPtr() {
  if (dptr != nullptr) {
    try {
      checkCudaErrors(cudaFree(dptr));
    } catch (...) {
      // empty
    }
  }
  dptr = nullptr;
  // if (residual_ptr != nullptr) {
  //   try {
  //     checkCudaErrors(cudaFree(residual_ptr));
  //   } catch (...) {
  //     // empty
  //   }
  // }
  // residual_ptr = nullptr;
}

} // namespace cuda
} // namespace at
